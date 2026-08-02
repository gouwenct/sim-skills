import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import cv2
import Utils as _U
import WindowF as _WF
import DataProcess as _DP
import scipy
from PyEMD import EMD
# from PyEMD.EMD import EMD

import cv2
import math
import datetime  #计时
import traceback
import warnings

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

tfft2 = torch.fft.fft2
tifft2 = torch.fft.ifft2
tfftn = torch.fft.fftn
tifftn = torch.fft.ifftn
tfftshift = torch.fft.fftshift
tifftshift = torch.fft.ifftshift


# =============================================================================
# GPU parameter-estimation policy (VRAM-safe single-candidate streaming)
# -----------------------------------------------------------------------------
# Hard constraints:
# - Controls live only in run_parameter['parameter_device'] = auto|cuda|cpu
# - auto: CUDA only if available and free VRAM >= 2.5 * estimated peak
# - cuda: must fail clearly (no silent fallback)
# - cpu: never touch CUDA
# - Transfer rationale remains VRAM safety: one candidate at a time, free FFT
#   intermediates immediately, never retain multiple candidate working sets.
# - Isolated same-shape probe (16,924,924) on 16GB RTX 4060 Ti:
#     original CPU-transfer path peak ~1667.5 MiB alloc / 1898 MiB reserved
#     GPU-expanded single-candidate path measured ~2292.9 MiB alloc / 2528 MiB reserved
#   Preflight uses the higher accelerated-path peak for safety.
# =============================================================================

_PEAK_PROBE_SHAPE = (16, 924, 924)
_PEAK_PROBE_ALLOC_MIB = 2292.9
_PEAK_PROBE_RESERVED_MIB = 2528.0
_VRAM_SAFETY_FACTOR = 2.5
_DEFAULT_PARAMETER_DEVICE = 'auto'
_ARRAY_RTOL = 1e-6
_ARRAY_ATOL = 1e-7


def _parameter_debug_output_enabled(opt=None, run_parameter=None):
    """Keep legacy diagnostics unless the copied entry requests essential output."""
    if isinstance(run_parameter, dict) and 'parameter_output_level' in run_parameter:
        level = run_parameter['parameter_output_level']
    else:
        level = getattr(opt, 'parameter_output_level', 'debug')
    return str(level).strip().lower() == 'debug'


def _cfai_png_output_enabled(opt=None, run_parameter=None):
    """Save modulation-depth/phase plots without enabling all debug artifacts."""
    if isinstance(run_parameter, dict) and 'parameter_output_level' in run_parameter:
        level = run_parameter['parameter_output_level']
    else:
        level = getattr(opt, 'parameter_output_level', 'debug')
    return str(level).strip().lower() in {'cfai_png', 'debug'}


def _log_pe(msg):
    print(f'[Para_Estimate_GPU] {msg}')


def estimate_parameter_peak_bytes(nz=None, nx=None, ny=None, shape=None):
    """Estimate peak allocated bytes for one candidate volume, scaled from probe."""
    if shape is not None:
        shape = tuple(int(v) for v in shape)
        if len(shape) == 2:
            nz, nx, ny = 1, shape[0], shape[1]
        else:
            nz, nx, ny = shape[0], shape[1], shape[2]
    else:
        nz = int(_PEAK_PROBE_SHAPE[0] if nz is None else nz)
        nx = int(_PEAK_PROBE_SHAPE[1] if nx is None else nx)
        ny = int(_PEAK_PROBE_SHAPE[2] if ny is None else ny)
    ref = float(_PEAK_PROBE_SHAPE[0] * _PEAK_PROBE_SHAPE[1] * _PEAK_PROBE_SHAPE[2])
    vol = float(max(1, nz * nx * ny))
    scale = max(1.0, vol / ref)  # never under-estimate below probe volume
    return int(_PEAK_PROBE_ALLOC_MIB * scale * 1024 * 1024)


def _cuda_mem_info_bytes(device_index=0):
    if not torch.cuda.is_available():
        return 0, 0
    free_b, total_b = torch.cuda.mem_get_info(device_index)
    return int(free_b), int(total_b)


def report_cuda_memory(tag='peak', device=None):
    """Report actual CUDA peak allocated/reserved memory (MiB)."""
    if not torch.cuda.is_available():
        _log_pe(f'{tag}: CUDA not available')
        return None
    if device is None:
        device = torch.device('cuda:0')
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != 'cuda':
        return None
    idx = device.index if device.index is not None else 0
    torch.cuda.synchronize(idx)
    alloc_mib = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
    reserved_mib = torch.cuda.max_memory_reserved(idx) / (1024 ** 2)
    cur_alloc_mib = torch.cuda.memory_allocated(idx) / (1024 ** 2)
    cur_reserved_mib = torch.cuda.memory_reserved(idx) / (1024 ** 2)
    free_b, total_b = _cuda_mem_info_bytes(idx)
    info = dict(
        peak_allocated_MiB=round(alloc_mib, 3),
        peak_reserved_MiB=round(reserved_mib, 3),
        current_allocated_MiB=round(cur_alloc_mib, 3),
        current_reserved_MiB=round(cur_reserved_mib, 3),
        free_MiB=round(free_b / (1024 ** 2), 3),
        total_MiB=round(total_b / (1024 ** 2), 3),
    )
    _log_pe(
        f'{tag}: peak_alloc={info["peak_allocated_MiB"]} MiB, '
        f'peak_reserved={info["peak_reserved_MiB"]} MiB, '
        f'cur_alloc={info["current_allocated_MiB"]} MiB, '
        f'free={info["free_MiB"]} MiB / total={info["total_MiB"]} MiB'
    )
    return info


def _release_cuda():
    """Synchronize after caller-owned candidate tensors have been deleted."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _normalize_device_mode(mode):
    if mode is None:
        return _DEFAULT_PARAMETER_DEVICE
    mode = str(mode).strip().lower()
    if mode not in ('auto', 'cuda', 'cpu'):
        raise ValueError(
            f"run_parameter['parameter_device'] must be auto|cuda|cpu, got {mode!r}"
        )
    return mode


def resolve_parameter_device(run_parameter=None, shape=None, opt=None):
    """
    Decide torch.device for parameter estimation.

    Returns (device, mode_requested, mode_effective, reason).
    """
    rp = run_parameter if isinstance(run_parameter, dict) else {}
    mode = _normalize_device_mode(rp.get('parameter_device', _DEFAULT_PARAMETER_DEVICE))

    # Shape heuristic from opt when available (MP: nz_re x 2nx x 2ny)
    if shape is None and opt is not None:
        try:
            nz = int(getattr(opt, 'nz_re', _PEAK_PROBE_SHAPE[0]))
            nx = int(getattr(opt, 'nx', _PEAK_PROBE_SHAPE[1] // 2) * 2)
            ny = int(getattr(opt, 'ny', _PEAK_PROBE_SHAPE[2] // 2) * 2)
            shape = (nz, nx, ny)
        except Exception:
            shape = _PEAK_PROBE_SHAPE
    if shape is None:
        shape = _PEAK_PROBE_SHAPE

    peak_bytes = estimate_parameter_peak_bytes(shape=shape)
    need_bytes = int(_VRAM_SAFETY_FACTOR * peak_bytes)

    if mode == 'cpu':
        return torch.device('cpu'), mode, 'cpu', 'parameter_device=cpu'

    if mode == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "parameter_device=cuda but torch.cuda.is_available() is False. "
                "No silent fallback."
            )
        free_b, total_b = _cuda_mem_info_bytes(0)
        # cuda mode: use CUDA even if tight, but fail clearly if free is tiny
        if free_b < peak_bytes:
            raise RuntimeError(
                f"parameter_device=cuda but free VRAM {free_b/1024**2:.1f} MiB "
                f"< estimated peak {peak_bytes/1024**2:.1f} MiB "
                f"(shape={tuple(shape)}, total={total_b/1024**2:.1f} MiB). "
                "No silent fallback."
            )
        reason = (
            f"parameter_device=cuda; free={free_b/1024**2:.1f} MiB, "
            f"peak_est={peak_bytes/1024**2:.1f} MiB, "
            f"safety_need={need_bytes/1024**2:.1f} MiB"
        )
        return torch.device('cuda:0'), mode, 'cuda', reason

    # auto
    if not torch.cuda.is_available():
        return torch.device('cpu'), mode, 'cpu', 'auto: CUDA not available'

    free_b, total_b = _cuda_mem_info_bytes(0)
    if free_b >= need_bytes:
        reason = (
            f"auto: free VRAM {free_b/1024**2:.1f} MiB >= "
            f"{_VRAM_SAFETY_FACTOR}×peak_est {peak_bytes/1024**2:.1f} MiB "
            f"(need {need_bytes/1024**2:.1f} MiB); use CUDA"
        )
        return torch.device('cuda:0'), mode, 'cuda', reason

    reason = (
        f"auto: free VRAM {free_b/1024**2:.1f} MiB < "
        f"{_VRAM_SAFETY_FACTOR}×peak_est {peak_bytes/1024**2:.1f} MiB "
        f"(need {need_bytes/1024**2:.1f} MiB); use CPU"
    )
    return torch.device('cpu'), mode, 'cpu', reason


def apply_parameter_device(opt, run_parameter=None, shape=None, reset_peak_stats=True):
    """Set opt.device from run_parameter policy. Returns resolution tuple."""
    device, mode_req, mode_eff, reason = resolve_parameter_device(
        run_parameter=run_parameter, shape=shape, opt=opt
    )
    opt.device = device
    if isinstance(run_parameter, dict):
        run_parameter['parameter_device_resolved'] = mode_eff
        run_parameter['parameter_device_reason'] = reason
    _log_pe(f'device={device} (requested={mode_req}, effective={mode_eff}); {reason}')
    if mode_eff == 'cuda' and reset_peak_stats and torch.cuda.is_available():
        idx = device.index if getattr(device, 'index', None) is not None else 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(idx)
    return device, mode_req, mode_eff, reason


def _np_to_device(arr, device, dtype=None):
    t = torch.as_tensor(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device=device, non_blocking=False)


def _as_numpy_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _peak_indices_match(a, b, name='array'):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, f'{name} shape mismatch {a.shape} vs {b.shape}'
    ia = np.unravel_index(int(np.argmax(np.abs(a))), a.shape)
    ib = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
    if ia != ib:
        return False, f'{name} discrete peak {ia} != {ib}'
    return True, f'{name} peak={ia}'


def compare_pgxy(cpu_pgxy, gpu_pgxy, rtol=_ARRAY_RTOL, atol=_ARRAY_ATOL):
    """Compare PGxy: selected positions exact; values also satisfy rtol/atol."""
    cpu = np.asarray(cpu_pgxy)
    gpu = np.asarray(gpu_pgxy)
    report = {'name': 'PGxy', 'shape_cpu': cpu.shape, 'shape_gpu': gpu.shape}
    if cpu.shape != gpu.shape:
        report.update(ok=False, reason=f'shape mismatch {cpu.shape} vs {gpu.shape}')
        return report
    position_ok = np.array_equal(cpu, gpu, equal_nan=True)
    abs_err = np.max(np.abs(cpu - gpu)) if cpu.size else 0.0
    rel_ok = np.allclose(cpu, gpu, rtol=rtol, atol=atol, equal_nan=True)
    report.update(
        ok=bool(position_ok and rel_ok),
        position_exact=bool(position_ok),
        max_abs_err=float(abs_err),
        allclose=bool(rel_ok),
        rtol=rtol,
        atol=atol,
    )
    if not report['ok']:
        report['reason'] = (
            f"PGxy parity fail: position_exact={position_ok}, allclose={rel_ok}, "
            f"max_abs_err={abs_err:.3e}"
        )
    return report


def compare_pgz(cpu_pgz, gpu_pgz, rtol=_ARRAY_RTOL, atol=_ARRAY_ATOL):
    """Compare PGz: selected positions exact; values also satisfy rtol/atol."""
    cpu = np.asarray(cpu_pgz)
    gpu = np.asarray(gpu_pgz)
    report = {'name': 'PGz', 'shape_cpu': cpu.shape, 'shape_gpu': gpu.shape}
    if cpu.shape != gpu.shape:
        report.update(ok=False, reason=f'shape mismatch {cpu.shape} vs {gpu.shape}')
        return report
    position_ok = np.array_equal(cpu, gpu, equal_nan=True)
    abs_err = np.max(np.abs(cpu - gpu)) if cpu.size else 0.0
    rel_ok = np.allclose(cpu, gpu, rtol=rtol, atol=atol, equal_nan=True)
    report.update(
        ok=bool(position_ok and rel_ok),
        position_exact=bool(position_ok),
        max_abs_err=float(abs_err),
        allclose=bool(rel_ok),
        rtol=rtol,
        atol=atol,
    )
    if not report['ok']:
        report['reason'] = (
            f"PGz parity fail: position_exact={position_ok}, allclose={rel_ok}, "
            f"max_abs_err={abs_err:.3e}"
        )
    return report


def compare_cfai(cpu_cfai, gpu_cfai, rtol=_ARRAY_RTOL, atol=_ARRAY_ATOL):
    """Compare cfai values with the agreed numerical tolerances."""
    cpu = np.asarray(cpu_cfai)
    gpu = np.asarray(gpu_cfai)
    report = {'name': 'cfai', 'shape_cpu': cpu.shape, 'shape_gpu': gpu.shape}
    if cpu.shape != gpu.shape:
        report.update(ok=False, reason=f'shape mismatch {cpu.shape} vs {gpu.shape}')
        return report
    abs_err = np.max(np.abs(cpu - gpu)) if cpu.size else 0.0
    rel_ok = np.allclose(cpu, gpu, rtol=rtol, atol=atol, equal_nan=True)
    report.update(
        ok=bool(rel_ok),
        max_abs_err=float(abs_err),
        allclose=bool(rel_ok),
        rtol=rtol,
        atol=atol,
    )
    if not report['ok']:
        report['reason'] = (
            f"cfai parity fail: allclose={rel_ok}, "
            f"max_abs_err={abs_err:.3e}"
        )
    return report


def compare_parameter_sets(cpu_pgxy, gpu_pgxy, cpu_pgz, gpu_pgz, cpu_cfai, gpu_cfai,
                           rtol=_ARRAY_RTOL, atol=_ARRAY_ATOL):
    """Compare PGxy/PGz/cfai packages. Returns (ok, reports)."""
    reports = [
        compare_pgxy(cpu_pgxy, gpu_pgxy, rtol=rtol, atol=atol),
        compare_pgz(cpu_pgz, gpu_pgz, rtol=rtol, atol=atol),
        compare_cfai(cpu_cfai, gpu_cfai, rtol=rtol, atol=atol),
    ]
    ok = all(r.get('ok', False) for r in reports)
    return ok, reports


def _run_with_device_fallback(opt, run_parameter, work_fn, tag='estimation'):
    """
    Run work_fn after applying device policy.
    auto mode: on GPU exception, log reason and fall back to CPU.
    cuda mode: re-raise (no silent fallback).
    cpu mode: CPU only.
    """
    device, mode_req, mode_eff, reason = apply_parameter_device(opt, run_parameter)
    try:
        result = work_fn()
        if mode_eff == 'cuda':
            report_cuda_memory(tag=f'{tag}-done', device=device)
        return result
    except Exception as e:
        if mode_req == 'auto' and mode_eff == 'cuda':
            _log_pe(
                f'{tag}: GPU path failed ({type(e).__name__}: {e}); '
                f'falling back to CPU. traceback follows:'
            )
            traceback.print_exc()
            if isinstance(run_parameter, dict):
                run_parameter['parameter_device_resolved'] = 'cpu'
                run_parameter['parameter_device_fallback'] = (
                    f'{type(e).__name__}: {e}'
                )
                run_parameter['parameter_device_reason'] = (
                    'auto: GPU exception; retrying this estimation on CPU'
                )
            opt.device = torch.device('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_pe(f'{tag}: retrying on CPU; requested mode remains auto.')
            return work_fn()
        if mode_req == 'cuda':
            _log_pe(f'{tag}: parameter_device=cuda failed; no silent fallback.')
        raise


def flip180(arr):#2D 3D都可以翻转
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr.copy()


def dft_corr2d(opt,h,g):# tt  h,g求的互相关(频域实现)
    """进行二维互相关"""
    g = torch.conj(g)   #先求共轭
    try:
        g = (flip180(g.numpy()))#再翻转180°
    except:
        g = (flip180(g.resolve_conj().numpy()))#
    k= np.array(h.shape)
    L = np.array(g.shape)

    g = _U.ToTensor(g)
    h = _U.ToTensor(h)
    f = torch.zeros_like(_U.ToTensor(k+L),dtype = opt.tdtype_com)
    f = tifftn( tfftn( h,s=tuple(k+L-1) )*tfftn( (g),s=tuple(k+L-1) ) )
    nx,ny = f.shape

    if nx!= (k+L)[0]: #就补一个零
        f = _U.pad_or_cut(f,x=(k+L)[0],y=(k+L)[1])
    return _U.ToNumpy(f)

def dft_corr3d(opt,h,g):# tt  h,g求的互相关(频域实现)
    """进行3维互相关
    将互相关变为卷积，再变到频域的乘积
    """
    g = torch.conj(g)   #先求共轭
    try:
        g = (flip180(g.numpy()))#再翻转180°
    except:
        g = (flip180(g.resolve_conj().numpy()))#

    k= np.array(h.shape)
    L = np.array(g.shape)

    g = _U.ToTensor(g)
    h = _U.ToTensor(h)
    f = torch.zeros_like(_U.ToTensor(k+L),dtype = opt.tdtype_com)
    f = tifftn(tfftn(h,s=tuple(k+L-1) )*tfftn((g),s=tuple(k+L-1)))
    nz,nx,ny = f.shape
    # print('dft_corr3d-f:',f.shape)
    
    if nx!= (k+L)[0]: #就补一个零
        f = _U.pad_or_cut(f,x=(k+L)[1],y=(k+L)[2],z=(k+L)[0])
    return _U.ToNumpy(f)


def Cir_Erode(data,size=3):
    """
    input: np
    output: np
    """
    if size == 0:
        return data
    kernel = np.ones((size, size), np.uint8)
    kernel_re = []
    rows, cols = kernel.shape
    for ii in range(rows):
        result = [0 if math.sqrt((ii-size//2)**2+(jj-size//2)**2) > size//2 else 1 for jj in range(cols)]
        kernel_re.append(result)
    kernel_re = np.array(kernel_re, np.uint8)

    data_erode =cv2.erode(data,kernel_re)


    if np.mean(data)!=0 and np.mean(data_erode) == 0:
        _U.save_tif(r'查看data.tif',data)
        _U.save_tif(r'查看data_erode.tif',data_erode)
        raise '3D 腐蚀太严重 ,需减少size'
    return data_erode


def Cir_Erode_3D(data,size=3):
    #没有轴向腐蚀，只是分成slice这样腐蚀
    data2=_U.image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_Erode(data[i],size=size)
    return data2


def saveparameter(path,PGxy,PGz,cfai,run_parameter):
    '''
    用于保存估计的参数
    '''
    path = str(path)
    filename = run_parameter['filename']
    Dr_sub = run_parameter['Dr_sub']
    pzz = str(run_parameter['pzzz'])
     
    if PGxy.any():
        filename1 = path+'\\'+'PGxy'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename1,PGxy)
        print(filename1)
        
        filename1 = _U.save_path('PGxy'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy',run_parameter)#保存在dataOut中
        np.save(filename1,PGxy)

    if PGz.any():
        filename2 = path+'\\'+'PGz'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename2,PGz)

        filename2 = _U.save_path('PGz'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy',run_parameter)#保存在dataOut中
        np.save(filename2,PGz)

    if cfai.any():
        filename3 = path+'\\'+'cfai'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename3,cfai)

        filename3 = _U.save_path('cfai'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy',run_parameter)#保存在dataOut中
        np.save(filename3,cfai)

def saveparameter02(path,cfai,run_parameter):
    '''
    用于保存估计的参数
    '''
    path = str(path)
    filename = run_parameter['filename']
    Dr_sub = run_parameter['Dr_sub']
    pzz = str(run_parameter['pzzz'])
    
    if cfai.any():
        filename3 = path+'\\'+'cfa02'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy'
        np.save(filename3,cfai)

        filename3 = _U.save_path('cfa02'+'-'+Dr_sub+'-'+filename+'-'+pzz+'.npy',run_parameter)#保存在dataOut中
        np.save(filename3,cfai)

def loadparameter(opt,path,run_parameter):
    '''
    用于加载估计的参数
    '''
    from pathlib import Path
    print(f'{path}')
    PGzF=0
    PGxyF=0
    cfaiF=0
    Dr_sub = run_parameter['Dr_sub']
    loop_file = Path(str(path)).iterdir()
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  #是文件夹
            if   ('PGxy' in ldr_i.name  ) :
                name = ldr_i.name
                print(f'加载的PGxy为：{name}')
                PGxy = np.load(ldr_i)
                PGxyF=1
            elif ('cfai' in ldr_i.name ) :
                name = ldr_i.name
                print(f'加载的cfai为：{name}')
                cfai = np.load(ldr_i)
                cfaiF=1
            elif ('PGz' in ldr_i.name ):
                name = ldr_i.name
                print(f'加载的PGz为：{name}')
                PGz = np.load(ldr_i)
                PGzF=1


    if PGxyF == 1:
        plong=0
        for i in range(opt.Nangles):
            for j in range(1,3):   #±2
                plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
            for j in range(3,opt.Norders):   #±1
                plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
        plong = plong/(opt.nc-3)


    if PGzF==0:
        PGz = np.zeros((3,7))
        print('加载pz参数失败！！！！')
    return cfai,PGxy,PGz,plong



def loadparameter2(opt,path,run_parameter):
    '''
    只加载了P值，若果是多层 还需要pz
    '''
    from pathlib import Path
    loop_file = Path(str(path)).iterdir()
    PGzF=0
    PGxyF=0
    Dr_sub = run_parameter['Dr_sub']
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  #是文件夹
            if   ('PGxy' in ldr_i.name):
                name = ldr_i.name
                print(f'加载的PGxy为：{name}')
                PGxy = np.load(ldr_i)
                PGxyF=1
            elif ('PGz' in ldr_i.name) :
                PGz = np.load(ldr_i)
                PGzF=1

    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Norders):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    
    if PGzF==0:
        PGz = np.zeros((3,7))
        print('加载pz参数失败！！！！')
    return PGxy,PGz,plong

def loadparameter3(opt,path):
    '''
    只加载了cfai值
    '''
    try:
        filename3 = path+'cfai.npy'
        cfai = np.load(filename3)
    except:
        print('加载参数失败！！！！')
        pass

    return cfai

def loadparameter4(opt,path,run_parameter):
    '''
    加载cfai02
    '''
    from pathlib import Path
    loop_file = Path(str(path)).iterdir()
    Dr_sub = run_parameter['Dr_sub']
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  #是文件夹
            if ('cfa02' in ldr_i.name ) :
                name = ldr_i.name
                print(f'加载的cfai02为：{name}')
                print(f'加载的cfai02为：{ldr_i}')
                cfai02 = np.load(ldr_i)
                return cfai02
    
    raise('No find cfai02')
    return 0


#############重新整理参数估计部分##########################
#int px计算
def get_intpxy(opt,ck0,ck1,i,j,methods='FAN'):
    '''
    input 3D
    '''
    assert len(ck0.shape) == 3
    assert len(ck1.shape) == 3
    nz,nx,ny = ck0.shape
    print(nz,nx,ny)
    # 范老师的方法
    if methods=='FAN':
        Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask2D,x=nx,y=ny))
        Hzuan = _U.image_reshape(Hzuan,dim=2)
        assert len(Hzuan.shape)==2
        
        cishu = abs(dft_corr2d(opt,Hzuan,Hzuan))     #消除面积效应
        if _parameter_debug_output_enabled(opt):
            _U.save_tif('cishu.tif',cishu.real)

        
        imagf0AC = ck0[nz//2]                         #取出中间一层/ projetction
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) #取出 相位角
        imagf1AC = ck1[nz//2]                         
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 
        
        
        if _parameter_debug_output_enabled(opt):
            _U.save_tif('imagf0_abs.tif',np.abs(imagf0_))
            _U.save_tif('imagf1_abs.tif',np.abs(imagf1_))
            _U.save_tif('imagf0_angle.tif',np.angle(imagf0_))
            _U.save_tif('imagf1_angle.tif',np.angle(imagf1_))
            _U.save_tif('imagf0AC.tif',imagf0AC.real)
        imagf0_ = _U.ToTensor(imagf0_)#*Hzuan        #消
        imagf1_ = _U.ToTensor(imagf1_)#*Hzuan

        jieguo = dft_corr2d(opt,(imagf0_),(imagf1_));     #用m=0阶 和m=2阶来求互相关
        if _parameter_debug_output_enabled(opt):
            _U.save_tif('jieguo.tif',jieguo.real)
        lihe = abs(jieguo/(cishu+opt.eps));           #消除面积影响
        
    elif methods=='SHAO':
        # Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=nz))
        # Hzuan = _U.image_reshape(Hzuan,dim=3)
        imagf0AC = ck0#*Hzuan                          #全部用上 
        imagf1AC = ck1#*Hzuan
        
        #变换到空域，先对imagf0AC横向增大
        imagf0AC_pad = _U.pad_or_cut(imagf0AC,x=2*nx,y=2*ny,z=1*nz)
        imagf1AC_pad = _U.pad_or_cut(imagf1AC,x=2*nx,y=2*ny,z=1*nz)
        # print(imagf0AC_pad.shape)
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      #变换到空域 
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      #
        # _U.save_tif('imagf0AC_kong.tif',np.real(imagf0AC_kong))

        lihe = (np.max( abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000,axis=0))   #轴向投影

    #通用部分，得到lihe后，确定最亮位置
    XV,YV=_U.meshgridxy(opt,kuodaxy=2)
    kr = np.sqrt(XV**2+YV**2)
    lihe[kr>(opt.pg+opt.fanwei)]=0
    lihe[kr<(opt.pg-opt.fanwei)]=0

    lihe = abs(lihe)

    # print((abs(lihe)==np.max(abs(lihe))).nonzero()) #这个为最大的值，坐标
    # print((lihe==np.max(lihe)).nonzero())    #找到最大值的坐标
    if _parameter_debug_output_enabled(opt):
        _U.save_tif(f"4 lihe-pxy-{i}-{j}.tif",lihe)
    opt.lihe2d.append(lihe)
    maxx = np.where(lihe==np.max(lihe))[0][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[1][0] # 127.5
    return maxx-opt.nx,maxy-opt.ny



def get_intpxy_2D(opt,ck0,ck1,i,j,methods='FAN'):
    '''
    input 2D
    '''
    assert len(ck0.shape) == 2
    assert len(ck1.shape) == 2
    nz,nx,ny = ck0.shape
    print(nz,nx,ny)
    # 范老师的方法
    if methods=='FAN':
        Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask2D,x=nx,y=ny))
        Hzuan = _U.image_reshape(Hzuan,dim=2)
        assert len(Hzuan.shape)==2
        
        cishu = abs(dft_corr2d(opt,Hzuan,Hzuan))     #消除面积效应
        if _parameter_debug_output_enabled(opt):
            _U.save_tif('cishu.tif',cishu.real)

        
        imagf0AC = ck0#[nz//2]                         #取出中间一层/ projetction
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) #取出 相位角
        imagf1AC = ck1#[nz//2]                         
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 
        if _parameter_debug_output_enabled(opt):
            _U.save_tif('imagf0AC.tif',imagf0AC.real)
        imagf0_ = _U.ToTensor(imagf0_)#*Hzuan        #消
        imagf1_ = _U.ToTensor(imagf1_)#*Hzuan

        jieguo = dft_corr2d(opt,(imagf0_),(imagf1_));     #用m=0阶 和m=2阶来求互相关
        if _parameter_debug_output_enabled(opt):
            _U.save_tif('jieguo.tif',jieguo.real)
        lihe = abs(jieguo/(cishu+opt.eps));           #消除面积影响

    elif methods=='SHAO':
        # Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=nz))
        # Hzuan = _U.image_reshape(Hzuan,dim=3)
        imagf0AC = ck0#*Hzuan                          #全部用上 
        imagf1AC = ck1#*Hzuan
        
        #变换到空域，先对imagf0AC横向增大
        imagf0AC_pad = _U.pad_or_cut(imagf0AC,x=2*nx,y=2*ny,z=1*nz)
        imagf1AC_pad = _U.pad_or_cut(imagf1AC,x=2*nx,y=2*ny,z=1*nz)
        # print(imagf0AC_pad.shape)
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      #变换到空域 
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      #
        # _U.save_tif('imagf0AC_kong.tif',np.real(imagf0AC_kong))

        lihe = (np.max( abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000,axis=0))   #轴向投影

    #通用部分，得到lihe后，确定最亮位置
    XV,YV=_U.meshgridxy(opt,kuodaxy=2)
    kr = np.sqrt(XV**2+YV**2)
    lihe[kr>(opt.pg+opt.fanwei)]=0
    lihe[kr<(opt.pg-opt.fanwei)]=0

    lihe = abs(lihe)

    # print((abs(lihe)==np.max(abs(lihe))).nonzero()) #这个为最大的值，坐标
    # print((lihe==np.max(lihe)).nonzero())    #找到最大值的坐标
    if _parameter_debug_output_enabled(opt):
        _U.save_tif(f"4 lihe-pxy-{i}-{j}.tif",lihe)
    opt.lihe2d.append(lihe)
    maxx = np.where(lihe==np.max(lihe))[0][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[1][0] # 127.5

    return maxx-opt.nx,maxy-opt.ny

#整数pz
def get_intpz(opt,ck0,ck1,i,j,methods='SHAO'):
    assert len(ck0.shape) == 3
    nz,nx,ny = ck0.shape
    
    #通用处理
    ck0 = ck0/np.max(abs(ck0))
    ck1 = ck1/np.max(abs(ck1))
    notch_filter1 = _WF.getotfAtt(opt,fwhm = 0.2*1000/opt.fc);

    fushi = 0
    # 范老师的方法
    if methods=='FAN':
        Hzuan = (_U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=nz)) #还得改回原来的3D
        Hzuan = _U.ToTensor(Cir_Erode(Hzuan,size = fushi))

        cishu = abs(dft_corr3d(opt,Hzuan,Hzuan))     #消除面积效应

        imagf0AC = ck0                     
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) #取出相位角
        imagf1AC = ck1                         
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 

        imagf0_ = _U.ToTensor(imagf0_)*Hzuan#*notch_filter1        #消除边缘影响
        imagf1_ = _U.ToTensor(imagf1_)*Hzuan#*notch_filter1

        jieguo = dft_corr3d(opt,(imagf0_),(imagf1_));     #用m=0阶 和m=2阶来求互相关
        lihe = abs(jieguo/(cishu+opt.eps));           #消除面积影响
        lihe = _U.pad_or_cut(lihe,x=nx,y=ny,z=2*nz)

    elif methods=='SHAO':
        #如果仅仅取出中间一层，效果不好，因为missing-corn
        #横向取中间一层OK,但轴向就不能取中间那么一层了
        #需要全部都计算一遍
        #然后投影取最大

        #变换到空域，先对imagf0AC轴向增大
        ck0_pad = _U.pad_or_cut(ck0,x=1*nx,y=1*ny,z=2*nz)
        ck1_pad = _U.pad_or_cut(ck1,x=1*nx,y=1*ny,z=2*nz)


        Hzuan = _U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=2*nz)
        Hzuan = Cir_Erode(Hzuan,size = fushi)
        imagf0AC_pad = ck0_pad * Hzuan * notch_filter1                         
        imagf1AC_pad = ck1_pad * Hzuan * notch_filter1
        
        # print(imagf0AC_pad.shape)
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      #变换到空域 
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      #
        # _U.save_tif('imagf0AC_kong.tif',np.real(imagf0AC_kong))
        lihe = abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000

    elif methods=='HIFI':
        ck0_pad = _U.pad_or_cut(ck0,x=1*nx,y=1*ny,z=2*nz)
        ck1_pad = _U.pad_or_cut(ck1,x=1*nx,y=1*ny,z=2*nz)

        Hzuan = _U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=2*nz)
        Hzuan = Cir_Erode(Hzuan,size = fushi)
        imagf0AC_pad = ck0_pad * Hzuan * notch_filter1                          #竖着取出中间一层 
        imagf1AC_pad = ck1_pad * Hzuan * notch_filter1
        
        #FFT
        imagf0AC_pad = fftn(imagf0AC_pad)
        imagf1AC_pad = fftn(imagf1AC_pad)
        imagf1AC_pad = imagf1AC_pad.conj() * (imagf0AC_pad)
        imagf1AC_pad = imagf1AC_pad/np.max(imagf1AC_pad)

        #ifft
        lihe = fftshift(ifftn(imagf1AC_pad))
        lihe = lihe*_U.pad_or_cut(opt.OTFmask3D,x=nx,y=ny,z=2*nz)
        

    XV,YV=_U.meshgridxy(opt,kuodaxy=1) # 这里不需要Z轴的信息
    kr = np.sqrt(XV**2+YV**2)
    lihe = abs(lihe/np.max(lihe));

    lihe[:,kr>(opt.pg+opt.fanwei)]=0
    lihe[:,kr<(opt.pg-opt.fanwei)]=0
    lihe[0:nz-opt.pgz] = 0    #估计一个z轴移动的范围
    lihe[nz+opt.pgz:] = 0     #去掉两边
    # lihe[nz-opt.fanweiz:nz+opt.fanweiz] = 0    #去掉最中间
    lihe[nz] = 0          #去掉最中间 

    lihe2 = (np.max( abs(lihe)/10,axis=2))  #这个只是看一下
    # lihe2[:,nx//2-3:nx//2+3]=0
    lihe2[nz,:] = 0

    opt.lihe3d.append(lihe)
    opt.lihe3dxy.append(lihe2)

    if _parameter_debug_output_enabled(opt):
        _U.save_tif(f"lihez-{i}-{j}-{methods}.tif",lihe)
    # _U.save_tif(f"lihez2-{i}-{j}-{methods}.tif",lihe2)
    
    maxz = np.where(lihe==np.max(lihe))[0][0]
    maxx = np.where(lihe==np.max(lihe))[1][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[2][0] # 127.5


    # print(f'methods:{methods}-整数{j}阶pz为',maxz-(opt.nz+2*opt.Zpadding))
    return int(maxz-(opt.nz_re/2)),int(maxx-opt.nx/2),int(maxy-opt.ny/2)  ##########这里多了个maxy



#确定，用什么作为更新依据
#调节x，y的机制是什么
#---------------------------------------------------------
#亚像素xy计算
def get_subpxy(opt,i,j,ck_0,ck_p,pgx,pgy,func):
    # 迭代互相关 COR
    #Lihe函数，作为更新依据
    xstep = 1
    ystep = 1
    buchangy = ystep
    buchangx = xstep
    jindu = 0.05
    # n_256 = opt.nx   #256
    PXY =np.array([pgx,pgy])/(2*opt.nx)
    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
    while( (buchangx>jindu) or (buchangy>jindu)):

        test = np.zeros(4)

        """max x方向"""
        maxx_tmp1 = pgx-1e-5; #上
        maxx_tmp2 = pgx+1e-5; #下
        
        for i in range(2):  #% 向上or 向下 （X轴往哪里移动）
            if i==0:
                kxtest=maxx_tmp1;
            else:
                kxtest=maxx_tmp2;
            PXY =np.array([kxtest,pgy])/(2*opt.nx)   #这里PXY的计算与图像大小有光，所以需要实时操作
        
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            test[i] = he1
        
        if((test[0]>test[1])):  #移动更小P 得到的结果更大
            flag_maxx=-1;       #recoder
        elif(test[0]<test[1]):
            flag_maxx=1;        #需要加大P
        else:
            try:
                flag_maxx=-flag_maxx
            except:
                flag_maxx=1

        while(buchangx>(jindu)):   #根据上面的上下指示，开始修正maxx_temp
            kxtest = pgx+flag_maxx*buchangx;      #不断的修正maxx_TMP

            PXY = np.array([kxtest,pgy])/(2*opt.nx)   #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1]) 
            if(he_tmp<=he_0):
                buchangx=buchangx*0.5
            else:
                print("updata pgx =",kxtest,'hetemp = ',he_tmp)
                pgx=kxtest           #更新！
                he_0=he_tmp
                #重置
                buchangy = ystep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy
            

        """max y方向"""
        maxy_tmp1 = pgy-1e-5; #左
        maxy_tmp2 = pgy+1e-5; #右
        for i in range(2,4):  #% 不同方向找最优
            if i==2:
                kytest=maxy_tmp1;
            else:
                kytest=maxy_tmp2;
            PXY = np.array([pgx,kytest])/(2*opt.ny)   #每次仅改变一个方向
        
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            test[i] = he1
        
        if((test[2]>test[3])):  #移动更小P 得到的结果更大
            flag_maxy=-1;       #recoder
        elif(test[2]<test[3]):
            flag_maxy=1;        #需要加大P
        else:
            try:
                flag_maxy=-flag_maxy
            except:
                flag_maxy=1
            
        
        while(buchangy>(jindu)):
            kytest = pgy+flag_maxy*buchangy;      #不断的修正maxx_TMP
            PXY = np.array([pgx,kytest])/(2*opt.ny)  #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            if(he_tmp<=he_0):
                buchangy=buchangy*0.5
            else:
                print("updata pgy =",kytest,'hetemp = ',he_tmp)
                pgy=kytest
                he_0=he_tmp
                #重置
                buchangx = xstep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy
    return pgx,pgy


def get_subpzxy_PCA(opt,ckp,j,pxint,pyint,pzint=0,flag='zy'):
    '''用PCA方式得到亚像素'''
    from numpy import linalg as la
    assert len(ckp.shape) == 3 
    #step 准备工作：
    
    #step2--int移频 先对ck_P移动
    ##法1
    # PXY = np.array([maxx-n_256,maxy-n_256])/(2*n_256)
    # ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    # ysh =(np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))   #单边移动 to beginning

    ##法2
    ckp_intmove = np.roll(ckp, (pzint,pxint,pyint) ,axis=(0,1,2) )


    #取mask
    if j<=2: #2th
        mask_size = 3
        padding_size= 5
        ROI = _U.pad_or_cut(ckp_intmove,x=mask_size, y =mask_size,z=mask_size)
        ROI = _U.pad_or_cut(ROI,x=mask_size+padding_size+1, y =mask_size+padding_size+1,z=opt.nz_re)
        #3D降2D
        ROI_XY = np.sum(ROI,axis=0)  #按Z进行累加,结果显示这个更加鲁棒
        Space_ROI = _U.ifftn(ROI_XY)

    #3D将2D 截取一面的方式
    if j>2: #1th
        mask_size = 3
        padding_size= 5
        ROI = _U.pad_or_cut(ckp_intmove,x=mask_size, y =mask_size,z=mask_size)
        ROI = _U.pad_or_cut(ROI,x=mask_size+padding_size+1, y =mask_size+padding_size+1,z=mask_size+padding_size+1)
        #3D降2D
        nzz,nxx,nyy = ROI.shape
        if flag == 'zy':
            ROI_ZY = np.sum(ROI,axis=1)
            # ROI_ZY = (ROI[:,nxx//2,:])
            # ROI_ZY = np.max(ROI,axis=1)
            Space_ROI = _U.ifftn(ROI_ZY)
        else:
            ROI_ZX = np.sum(ROI,axis=2)
            # ROI_ZX = (ROI[:,:,nyy//2])
            # ROI_ZX = np.max(ROI,axis=2)
            Space_ROI = _U.ifftn(ROI_ZX)



    #step3 计算逆变换，获得指数项
    Phase_ROI = np.exp(1j*np.angle(Space_ROI))


    #step4 SVD分解 得到U  V
    assert len(Phase_ROI.shape)==2
    u,sigma,vt = la.svd(Phase_ROI)   #进行SVD时 需要是2D
    
    #step5 最小二乘法拟合
    dx_z,u[:,0] = Eigenvector_Fitting(u[:,0],j)
    dy,vt[0,:] = Eigenvector_Fitting(vt[0,:],-j)

    #计算拟合出来的初相
    # sigma[1:]=0   
    # Phase_ROI_new = (u @ np.diag(sigma)) @ vt
    return dx_z,dy

def Eigenvector_Fitting(eigenone,j):
    #eigenone 对u来说是[:,0]；对v来说是[0,:]
    #j 用来表示这是第几阶, +表示u  -表示v
    Eigenvector_angle= np.unwrap(np.angle(eigenone))  #取angle 并消除-pi pi的区别
    Eigenvector_angle_fit = Eigenvector_angle[1:-1].reshape(-1)  #因为有padding 去掉头尾不准确的值
    cai_size = Eigenvector_angle_fit.shape[0]         #长度

    FIT = np.polyfit(np.arange(cai_size), Eigenvector_angle_fit, 1) # 用1次多项式拟合，依次为系数1，最后为bias
    sub = -FIT[0]*cai_size/(2*np.pi)                  #计算亚像素

    if sub>=1:
        _U.save_tif(f'Eigenvector_angle-{j}.tif',Eigenvector_angle)
        print(f'sub：{sub}有问题！请查看整数像素，请修改mask窗口大小')
    else:
        if abs(j)<3:  #2阶
            if j>0:  
                print('dx=',sub)   #还需要判断，这个值应该小于1
                                   #在zy中，2阶z的改变量很小 可以相信
            elif j<0:
                print('dy=',sub)
        elif abs(j)>=3:
            if j>0:  
                print('dz=',sub)   #还需要判断，这个值应该小于1
                                   #在zy中，2阶z的改变量很小 可以相信
            elif j<0:
                print('检验dy/dx=',sub) #这个dy来用判断的 不参与更新

    #拟合出理想的Eigenvector
    New_angle = np.polyval(FIT,np.arange(eigenone.shape[0]))
    # _U.save_tif('3d-u(new)_angle.tif',New_U_angle)
    # u[:,0] = New_U_angle
    return sub,New_angle




def get_subpzxy_PCA01(opt,ckp,j,pxint,pyint,pzint=0,flag='zy'):
    '''用PCA方式得到亚像素，0级与±1级光 '''
    from numpy import linalg as la
    assert len(ckp.shape) == 3 
    #step 准备工作：
    
    #step2--int移频 先对ck_P移动
    ##法1
    # PXY = np.array([maxx-n_256,maxy-n_256])/(2*n_256)
    # ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    # ysh =(np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))   #单边移动 to beginning

    ##法2
    ckp_intmove = np.roll(ckp, (pzint,pxint,pyint) ,axis=(0,1,2) )


    #取mask
    # if j<=2: #2th
    #     mask_size = 3
    #     padding_size= 5
    #     ROI = _U.pad_or_cut(ckp_intmove,x=mask_size, y =mask_size,z=mask_size)
    #     ROI = _U.pad_or_cut(ROI,x=mask_size+padding_size+1, y =mask_size+padding_size+1,z=opt.nz_re)
    #     #3D降2D
    #     ROI_XY = np.sum(ROI,axis=0)  #按Z进行累加,结果显示这个更加鲁棒
    #     Space_ROI = _U.ifftn(ROI_XY)

    #3D将2D 截取一面的方式
    if j<=2: #1th
        mask_size = 3
        padding_size= 5
        ROI = _U.pad_or_cut(ckp_intmove,x=mask_size, y =mask_size,z=mask_size)
        ROI = _U.pad_or_cut(ROI,x=mask_size+padding_size+1, y =mask_size+padding_size+1,z=mask_size+padding_size+1)
        #3D降2D
        nzz,nxx,nyy = ROI.shape
        if flag == 'zy':
            ROI_ZY = np.sum(ROI,axis=1)
            # ROI_ZY = (ROI[:,nxx//2,:])
            # ROI_ZY = np.max(ROI,axis=1)
            Space_ROI = _U.ifftn(ROI_ZY)
        else:
            ROI_ZX = np.sum(ROI,axis=2)
            # ROI_ZX = (ROI[:,:,nyy//2])
            # ROI_ZX = np.max(ROI,axis=2)
            Space_ROI = _U.ifftn(ROI_ZX)



    #step3 计算逆变换，获得指数项
    Phase_ROI = np.exp(1j*np.angle(Space_ROI))


    #step4 SVD分解 得到U  V
    assert len(Phase_ROI.shape)==2
    u,sigma,vt = la.svd(Phase_ROI)   #进行SVD时 需要是2D
    
    #step5 最小二乘法拟合
    dx_z,u[:,0] = Eigenvector_Fitting(u[:,0],j)
    dy,vt[0,:] = Eigenvector_Fitting(vt[0,:],-j)

    #计算拟合出来的初相
    # sigma[1:]=0   
    # Phase_ROI_new = (u @ np.diag(sigma)) @ vt
    return dx_z,dy


#第一种更新策略 等分pz1为左右10等份
def get_subpz(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func,jindu=0.1,fanwei=6):
    #方法1，等分法
    nz = (opt.nz+2*opt.Zpadding) #
    nx = opt.nx  #

    PXY = np.array([pgx,pgy])/(2*nx)
    print('ck_p.shape',ck_p.shape)
    print('nz=',nz)
    PGZ =(pgz)/(nz*2)

    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PGZ,m)
    fanwei=fanwei # 一边分为fanwei-1 怎么多份 ,一边= jindu*（fanwei-1） 那么多长度
    he_array = np.zeros(2*fanwei-1)
    he_array[fanwei-1] = he_0
    print('标准为：',he_0.round(6))
    for ii in range(1,fanwei):
        #+i倍精度右
        print('+',ii,' ',pgz+ii*jindu)
        PGZ=(pgz+ii*jindu)/(nz*2)
        # he_array[(fanwei-1)+i] = LiheZ3D2(opt,PXY[0],PXY[1],pgzz,m,kuodaz=kuodaz)
        temp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PGZ,m)
        print(temp.round(6))
        he_array[(fanwei-1)+ii] = temp
    # print('右边的数值')
    # print(he_array)
    for ii in range(1,fanwei):
        #-减去i倍精度 左
        print('-',ii,' ',pgz-ii*jindu)
        PGZ=(pgz-ii*jindu)/(nz*2)
        # he_array[(fanwei-1)-i] = LiheZ3D2(opt,PXY[0],PXY[1],pgzz,m,kuodaz=kuodaz)
        temp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PGZ,m)
        he_array[(fanwei-1)-ii] = temp
        print(temp.round(6))
    # print('左边的数值')
    print(he_array)
    #画图

    # X = np.arange(-(fanwei-1)*jindu+pgz,fanwei*jindu+pgz,jindu)
    X = np.linspace(-(fanwei-1)*jindu+pgz,(fanwei-1)*jindu+pgz,fanwei*2-1)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X , he_array, color = 'b', width = jindu/2)
    plt.xlim(-(fanwei-1)*jindu+pgz,(fanwei-1)*jindu+pgz)
    # plt.show()

    j = np.where(he_array==np.max(he_array))[0][0]
    
    if j>(fanwei-1): # 6,7,8,9,10 加上了一个数
        print('j 加上',( j-(fanwei-1) ))
        print(pgz+( j-(fanwei-1) )*jindu)
        return pgz+( j-(fanwei-1) )*jindu
    else: #j>0 and j<=5: #左边为减去一个数
        print('j 减去',((fanwei-1)-j ))
        print(pgz- ((fanwei-1)-j )*jindu)
        return pgz-((fanwei-1)-j )*jindu

#第二种更新策略:xyz
def get_subpz2(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func):

    '只更新z方向'
    zstep = 0.1   #因为只能朝一个方向前进，所以这个值要小
    xstep = 0.1
    ystep = 0.1
    buchangz = zstep
    buchangy = ystep
    buchangx = xstep
    jindu = 0.025
    step_test = 1e-1
    nz_256 = (opt.nz+2*opt.Zpadding) #opt.nz   #10
    nx_256 = opt.nx  #175
    # maxx = maxx
    # maxy = maxy
    # maxz = maxz
    # print('nz_256',nz_256)
    # print('nx_256',nx_256)
    
    PXY = np.array([pgx,pgy])/(2*nx_256)
    PZ =(pgz)/(2*nz_256)
    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)


    #x判断方向
    maxx_tmp1 = pgx-1e-4; #上
    maxx_tmp2 = pgx+1e-4; #下
    test = np.zeros(2)
    for i in range(2):  #% 向上or 向下 （X轴往哪里移动）
        if i==0:
            kxtest=maxx_tmp1;
        else:
            kxtest=maxx_tmp2;
        PXY =np.array([kxtest,pgy])/(2*nx_256)   #这里PXY的计算与图像大小有光，所以需要实时操作
    
        test[i] = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)                                                                      #计算sum 比值
        
        if((test[0]>test[1])):  #移动更小P 得到的结果更大
            flag_maxx=-1;       #recoder
        elif(test[0]<test[1]):
            flag_maxx=1;        #需要加大P

    #y判断方向
    maxy_tmp1 = pgy-1e-4; #左
    maxy_tmp2 = pgy+1e-4; #右
    for i in range(2):  #% 不同方向找最优
        if i==0:
            kytest=maxy_tmp1;
        else:
            kytest=maxy_tmp2;
        PXY = np.array([pgx,kytest])/(2*nx_256)   #每次仅改变一个方向
        test[i] = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
    
    if((test[0]>test[1])):  #移动更小P 得到的结果更大
        flag_maxy=-1;       #recoder
    elif(test[0]<test[1]):
        flag_maxy=1;        #需要加大P

    #z判断方向
    maxz_tmp1 = pgz-step_test; #
    maxz_tmp2 = pgz+step_test; #
    PXY = np.array([pgx,pgy])/(2*nx_256)  #目前最好的maxx

    for i in range(2):  #% 向上or 向下 
        if i==0:
            kztest=maxz_tmp1;
        else:
            kztest=maxz_tmp2; #1向上
        PZ =(kztest)/(2*nz_256)   #这里PXY的计算与图像大小有光，所以需要实时操作
        test[i] = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)                                                         #计算sum 比值
    
    if((test[0]>test[1])):  #移动更小P 得到的结果更大
        flag_maxz=-1;       #recoder
    elif(test[0]<test[1]):
        flag_maxz=1;        #需要加大P

    while( (buchangz>jindu)  or (buchangx>jindu) or (buchangy>jindu) ):
        """max x方向"""
        while(buchangx>(jindu)):   #根据上面的上下指示，开始修正maxx_temp
            maxx_tmp = pgx+flag_maxx*buchangx;      #不断的修正maxx_TMP
            kxtest=maxx_tmp;
            PXY = np.array([kxtest,pgy])/(2*nx_256)   #每次仅改变一个方向
            # he_tmp = LiheZ3D(opt,PXY[0],PXY[1],PZ,m)
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)  
            if(he_tmp<=he_0):
                #sj = random.randint(-1,1)
                #buchangx=(1+sj/9)*buchangx*0.5
                buchangx=buchangx*0.5
            else:
                print("updata maxx,maxx=",maxx_tmp,'hetemp = ',he_tmp)
                pgx=maxx_tmp
                he_0=he_tmp
                #一旦一个方向移动了，就重置其他方向的步长
                buchangz = 0.1
                buchangy = 0.1
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找max
            
        while(buchangy>(jindu)):
            maxy_tmp = pgy+flag_maxy*buchangy;      #不断的修正maxx_TMP
            kytest = maxy_tmp;
            PXY = np.array([pgx,kytest])/(2*nx_256)
              #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
            if(he_tmp<=he_0):
                #sj = random.randint(-1,1)
                #buchangy=(1+sj/9)*buchangy*0.5
                buchangy = buchangy*0.5
            else:
                print("updata pgx=",maxy_tmp,'hetemp = ',he_tmp)
                pgy = maxy_tmp
                he_0 = he_tmp
                #重置
                buchangz = 0.1
                buchangx = 0.1
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy

        """max z方向"""
        PXY = np.array([pgx,pgy])/(2*nx_256)  #目前最好的maxx
        while(buchangz>(jindu)):   #根据上面的上下指示，开始修正maxx_temp
            print('z方向为',flag_maxz)
            maxz_tmp = pgz+flag_maxz*buchangz;      #不断的修正maxx_TMP
            kztest=maxz_tmp;
            PZ = (kztest)/(2*nz_256)   #这里PXY的计算与图像大小有关，所以需要实时操作
            # he_tmp = LiheZ3D(opt,PXY[0],PXY[1],PZ,m)
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)

            if(he_tmp<=he_0):
                #sj = random.randint(-1,1)
                #buchangx=(1+sj/9)*buchangx*0.5
                buchangz=buchangz*0.5
            else:
                print("updata maxz",maxz_tmp,'hetemp = ',he_tmp)
                pgz=maxz_tmp
                he_0=he_tmp
                buchangx = 0.1
                buchangy = 0.1
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环
        print(f'buchangx={buchangx},buchangy={buchangy},buchangz={buchangz}')
    return pgz,pgx,pgy

def get_subpz2_2(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func):
    #这个相对于上面，运行速度更慢，因为在各个方向上拟合
    zstep = 0.5   #因为只能朝一个方向前进，所以这个值要小
    xstep = 0.5
    ystep = 0.5
    buchangz = zstep
    buchangy = ystep
    buchangx = xstep
    jindu = 0.025
    step_test = 1e-1             #用来尝试
    nz_256 = (opt.nz+2*opt.Zpadding) #opt.nz   #10
    nx_256 = opt.nx  #175

    PXY =np.array([pgx,pgy])/(2*nx_256)
    PZ =(pgz)/(2*nz_256)
    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)

    while( (buchangx>jindu) or (buchangy>jindu) or (buchangz>jindu)):
        """max x方向"""
        maxx_tmp1 = pgx-step_test; #上
        maxx_tmp2 = pgy+step_test; #下
        PZ = (pgz)/(2*nz_256)
        test = np.zeros(6)

        for i in range(2):  #% 向上or 向下 （X轴往哪里移动）
            if i==0:
                kxtest=maxx_tmp1;
            else:
                kxtest=maxx_tmp2;
                
            PXY = np.array([kxtest,pgy])/(2*nx_256)   #这里PXY的计算与图像大小有光，所以需要实时操作
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m); #he(ck_2d_center_pad,Om_2D_center2_pad,Om_2D_center_pad,PXY) #将m=0与m=2 进行冗余区域求解
                                                                              #计算sum 比值
            test[i] = he1
        
        if((test[0]>test[1])):  #移动更小P 得到的结果更大
            flag_maxx=-1;       #recoder
        elif(test[0]<=test[1]):
            flag_maxx=1;        #需要加大P

        while(buchangx>(jindu)):   #根据上面的上下指示，开始修正maxx_temp
            maxx_tmp = pgx+flag_maxx*buchangx;      #不断的修正maxx_TMP
            kxtest=maxx_tmp;
            PXY = np.array([kxtest,pgy])/(2*nx_256)   #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
            if(he_tmp<=he_0):
                #sj = random.randint(-1,1)
                #buchangx=(1+sj/9)*buchangx*0.5
                buchangx=buchangx*0.5
            else:
                print("updata pgx =",maxx_tmp,'hetemp = ',he_tmp)
                pgx=maxx_tmp
                he_0=he_tmp
                #重置
                buchangy = ystep
                buchangz = zstep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy
        
        """max y方向"""
        maxy_tmp1 = pgy-step_test; #左
        maxy_tmp2 = pgy+step_test; #右
        PZ = (pgz)/(2*nz_256)
        for i in range(2,4):  #% 不同方向找最优
            if i==2:
                kytest=maxy_tmp1;
            else:
                kytest=maxy_tmp2;
            PXY = np.array([pgx,kytest])/(2*nx_256)   #每次仅改变一个方向
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
            test[i] = he1
        
        if((test[2]>test[3])):  #移动更小P 得到的结果更大
            flag_maxy=-1;       #recoder
        elif(test[2]<=test[3]):
            flag_maxy=1;        #需要加大P
            
        
        while(buchangy>(jindu)):
            maxy_tmp = pgy+flag_maxy*buchangy;      #不断的修正maxx_TMP
            kytest=maxy_tmp;
            PXY = np.array([pgx,kytest])/(2*nx_256)  #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
            if(he_tmp<=he_0):
                #sj = random.randint(-1,1)
                #buchangy=(1+sj/9)*buchangy*0.5
                buchangy=buchangy*0.5
            else:
                print("updata maxy,maxy=",maxy_tmp,'hetemp = ',he_tmp)
                pgy=maxy_tmp
                he_0=he_tmp
                #重置
                buchangx = xstep
                buchangz = zstep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy

        """max z方向"""
        maxz_tmp1 = pgz-step_test; #左
        maxz_tmp2 = pgz+step_test; #右
        PXY = np.array([pgx,pgy])/(2*nx_256)

        for i in range(4,6):  #% 不同方向找最优
            if i==4:
                kztest=maxz_tmp1
            else:
                kztest=maxz_tmp2
            PZ =(kztest)/(2*nz_256)
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
            test[i] = he1
        
        if((test[4]>test[5])):  #移动更小P 得到的结果更大
            flag_maxz=-1;       #recoder
        elif(test[4]<=test[5]):
            flag_maxz=1;        #需要加大P

        while(buchangz>(jindu)):
            maxz_tmp = pgz+flag_maxz*buchangz;      #不断的修正maxx_TMP
            kztest = maxz_tmp
              #每次仅改变一个方向
            PZ = (kztest)/(2*nz_256)
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1],PZ,m)
            if(he_tmp<=he_0):
                #sj = random.randint(-1,1)
                #buchangy=(1+sj/9)*buchangy*0.5
                buchangz=buchangz*0.5
            else:
                print("updata maxz",maxz_tmp,'hetemp = ',he_tmp)
                pgz=maxz_tmp
                he_0=he_tmp
                buchangx = xstep
                buchangy = ystep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy
    return pgz,pgx,pgy


#pz的更新依据 综合pz oxy
#2023-10-23 解耦liheXYZ = liheXYZ + cal_c_fai  代码
def LiheXYZ(opt,i,j,ck_0,ck_p,kx,ky,kz=0,m=2,cfai=0,mask_ger=0,run_parameter=''):
    "2D平面计算/3D计算都可以,用flag=0（2d）区分"
    "传入m=1 2  是用面估计"
    "传入m>2时，使用体估计"

    if m>2 and opt.flag_N=='MP' and opt.Norders==7 : #±1阶 3D
        flag=1 #根据传入的m 自动判断使用3D 还是2D

    if j <3 and cfai!=4: #2阶P亚像素求解 使用中间层面估计
        piancha = 0
        flag= 0
        filename='' ##不用保存

    elif j>=3 and cfai!=4: #1阶pz亚像素求解，使用体估计
        piancha = 0 #run_parameter['piancha']
        flag= 1 #run_parameter['flagV']
        filename=''#run_parameter['filename']

    else:
        piancha = 0#run_parameter['piancha']
        flag= 1#run_parameter['flagV']
        filename = ''#run_parameter['filename']


    assert len(ck_0.shape)==3

    nz,nx,ny = ck_0.shape
    device = getattr(opt, 'device', torch.device('cpu'))
    if isinstance(device, str):
        device = torch.device(device)

    cache = getattr(opt, '_lihe_scalar_cache', None)
    cache_key = None
    if isinstance(cache, dict):
        ck_0_array = np.asarray(ck_0)
        ck_p_array = np.asarray(ck_p)
        cache_key = (
            device.type,
            device.index,
            flag,
            ck_0_array.__array_interface__['data'][0],
            ck_p_array.__array_interface__['data'][0],
            ck_0_array.shape,
            ck_p_array.shape,
            ck_0_array.strides,
            ck_p_array.strides,
            float(kx),
            float(ky),
            float(kz),
        )
        if cache_key in cache:
            opt._lihe_scalar_cache_hits += 1
            return cache[cache_key]
        opt._lihe_scalar_cache_misses += 1

    if flag==0: #±2阶使用平面2D--------------#2D
        XV,YV = _U.meshgridxy(opt,kuodaxy=2)
        ysh = np.exp(2j*np.pi*(kx*XV+ky*YV))
        #[1]0阶 【image】
        imgf0AC = ck_0[nz//2+piancha,:,:]  #0 提取中间一层
        imgf0 = imgf0AC/(abs(imgf0AC)+opt.eps)   #相位项1
        imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=1)    #二维情况下加pading 2X
        
        #[2]0阶 【otf、otfmask】2d
        otf = opt.OTF2D
        otfmask = opt.OTFmask2D

        #[3]P阶【image】
        imgfmAC = ck_p[nz//2+piancha,:,:]
        imgfm = imgfmAC/(abs(imgfmAC)+opt.eps)   #相位项1
        imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=1)    # Dr 2x
        
        #[4] P阶【OTF_move、OTFmask_move】2d
        # _ , otfmask_move,_ = _DP.otf_mask_3d_2d(opt,PGxy =[kx*(2*opt.nx),ky*(2*opt.ny)],PGz = [kz*opt.nz_re], Debug=0) #
        # otfmask_move = Cir_Erode(otfmask_move,size=0) #2D 腐蚀一下

    else: #±1阶 且 MP 使用3D估计法----------------#3D
        ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
        ysh = np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV))
        #[1] 0阶 【image】
        imgf0AC = ck_0  
        imgf0 = imgf0AC/(abs(imgf0AC)+opt.eps)   #相位项
        imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=opt.nz_re)    #二维情况下加pading 2X
        
        #[2] 0阶 【otf】3d
        otf = _U.image_reshape(opt.OTF,dim=3)
        otfmask = opt.OTFmask3D

        #[3] P阶【image】
        imgfmAC = ck_p
        imgfm = imgfmAC/(abs(imgfmAC)+opt.eps)  #提取为相位角
        imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=opt.nz_re)    # Dr 2x
        
        #[4] P阶【OTF_move、OTFmask_move】
        # otfmask_move , _ = _DP.otf_mask_3d_2d(opt,PGxy =[kx*(2*opt.nx),ky*(2*opt.ny)],PGz = [kz*opt.nz_re], Debug=0) #
        # otfmask_move = Cir_Erode_3D(otfmask_move,size=3)   #3D 腐蚀一下
        

    
    #################GPU加速，移动（single-candidate, VRAM-safe）##############
    # Rationale: keep FFT + overlap reduction on device so only a scalar returns.
    # Never retain multiple candidate working sets; free FFT intermediates ASAP.
    ysh_t = _np_to_device(ysh, device)
    otf_t = _np_to_device(otf, device)
    otfmask_t = _np_to_device(otfmask, device)
    imgfm_t = _np_to_device(imgfm, device)
    imgf0_t = _np_to_device(imgf0, device)

    #[3.]imgf P阶 移动
    imgfm_sp = _U.tifftn(imgfm_t)
    imgfm_move = _U.tfftn(imgfm_sp * ysh_t)
    del imgfm_sp

    #[4.]otf_move 生成
    otf_sp = _U.tifftn(otf_t)
    psf_ = tfftshift(otf_sp).real
    del otf_sp
    otf_move = (_U.tfftn(tifftshift(psf_ * ysh_t))).abs()
    del psf_

    #[]otfmask_move 生成
    otfmask_sp = _U.tifftn(otfmask_t)
    psf_mask_ = tfftshift(otfmask_sp).real
    del otfmask_sp
    otfmask_move = (_U.tfftn(tifftshift(psf_mask_ * ysh_t))).abs()
    del psf_mask_
    otfmask_move = torch.where(
        otfmask_move > 0.9,
        torch.ones_like(otfmask_move),
        torch.zeros_like(otfmask_move),
    )

    # 交叉区域（on-device reduction → scalar only transfer）
    w0 = (imgf0_t * otfmask_t) * (otf_move * otfmask_move)
    w1 = (otf_t * otfmask_t) * (imgfm_move * otfmask_move)
    w0 = w0.to(torch.complex64)
    w1 = w1.to(torch.complex64)
    overlap_mask = otfmask_move * otfmask_t
    overlap_sum = torch.abs(torch.sum(w1 * torch.conj(w0)))
    overlap_mask_sum = torch.abs(torch.sum(overlap_mask))
    a = _as_numpy_scalar(overlap_sum / overlap_mask_sum)

    # Explicitly release this candidate's working set
    del (
        ysh_t, otf_t, otfmask_t, imgfm_t, imgf0_t,
        imgfm_move, otf_move, otfmask_move, w0, w1, overlap_mask,
        overlap_sum, overlap_mask_sum,
    )
    _release_cuda()
    result = abs(a)
    if cache_key is not None:
        cache[cache_key] = result
    return result        #方法1

def cal_c_fai(opt,i,j,ck_0,ck_p,kx,ky,kz=0,filename='',run_parameter=""):
    "引入计算cfai函数"
    #这一步均使用体估计
    assert len(ck_0.shape)==3
    nz,nx,ny = ck_0.shape
    #空域移动矩阵(已扩大)
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    ysh  = (np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))   #全部移动
 
    #[1] 0阶ck
    imgf0 = np.copy(ck_0)
    imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=opt.nz_re)

    #[2] 0阶OTFmask
    otf = abs(_U.image_reshape(opt.OTF,dim=3))
    otfmask = opt.OTFmask3D

    #[3] P阶ck
    imgfm = np.copy(ck_p)
    imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=opt.nz_re)    # Dr 2x
    
    ################GPU加速，移动（single-candidate, VRAM-safe）#############
    # Keep FFT move + preliminary products on device; download only arrays
    # still required by CPU-only Cir_Erode / EMD. Free FFT intermediates.
    device = getattr(opt, 'device', torch.device('cpu'))
    if isinstance(device, str):
        device = torch.device(device)

    ysh_t = _np_to_device(ysh, device)
    otf_t = _np_to_device(otf, device)
    otfmask_t = _np_to_device(otfmask, device)
    imgfm_t = _np_to_device(imgfm, device)
    imgf0_t = _np_to_device(imgf0, device)

    #[3.]imgf P阶 移动
    imgfm_sp = _U.tifftn(imgfm_t)
    imgfm_move_t = _U.tfftn(imgfm_sp * ysh_t)
    del imgfm_sp

    #[4.]otf_move 生成
    otf_sp = _U.tifftn(otf_t)
    psf_ = tfftshift(otf_sp).real
    del otf_sp
    otf_move_t = (_U.tfftn(tifftshift(psf_ * ysh_t))).abs()
    del psf_

    #[]otfmask_move 生成
    otfmask_sp = _U.tifftn(otfmask_t)
    psf_mask_ = tfftshift(otfmask_sp).real
    del otfmask_sp
    otfmask_move_t = (_U.tfftn(tifftshift(psf_mask_ * ysh_t))).abs()
    del psf_mask_
    otfmask_move_t = torch.where(
        otfmask_move_t > 0.9,
        torch.ones_like(otfmask_move_t),
        torch.zeros_like(otfmask_move_t),
    )

    # On-device products that need CPU for Cir_Erode / EMD only
    w0_t = (imgf0_t * otfmask_t) * (otf_move_t * otfmask_move_t)
    w1_t = (otf_t * otfmask_t) * (imgfm_move_t * otfmask_move_t)
    overlap_mask_t = otfmask_move_t * otfmask_t

    w0 = w0_t.detach().cpu().numpy()
    w1 = w1_t.detach().cpu().numpy()
    overlap_mask = overlap_mask_t.to(dtype=torch.uint8).detach().cpu().numpy()

    # Release GPU candidate working set before CPU EMD (VRAM safety)
    del (
        ysh_t, otf_t, otfmask_t, imgfm_t, imgf0_t,
        imgfm_move_t, otf_move_t, otfmask_move_t,
        w0_t, w1_t, overlap_mask_t,
    )
    _release_cuda()
    #####################移动完成取消########################################

    #mask 缩小 (CPU / OpenCV; unchanged physical convention)
    overlap_mask = _U.Cir_Erode_3D(
        np.ascontiguousarray(overlap_mask, dtype=np.uint8),
        size=int(opt.fc//5),
    )
    

    #两者的比值
    overlap_cfai_3D = (w1/(w0+opt.eps))   * overlap_mask.astype(np.bool_)
    overlap_cfai = np.copy(overlap_cfai_3D) #

    # 复数线性回归
    overlap_c    = np.abs(overlap_cfai)
    overlap_fai  = np.angle(overlap_cfai/(abs(overlap_cfai)+opt.eps)); 

    #记录重叠区域
    if _parameter_debug_output_enabled(opt, run_parameter):
        opt.abscm3d.append(overlap_c)
        opt.angcm3d.append(overlap_fai)
        opt.overlap_mask3d.append(overlap_mask)

    if _parameter_debug_output_enabled(opt, run_parameter):
        _U.save_tif(f'overlap_mask-{i,j}.tif',overlap_mask)
        _U.save_tif(f'overlap_c-{i,j}.tif',overlap_c)
        _U.save_tif(f'overlap_fai{i,j}.tif',overlap_fai)
    # asd=asd
    
    'EMD'
    print('————————cal_c_fai 为 体估计——————————')
    overlap_c_1d = overlap_c.flatten()[np.argwhere(overlap_c.flatten())].squeeze()
    overlap_fai_1d = overlap_fai.flatten()[np.argwhere(overlap_fai.flatten())].squeeze()
    ######################
    jdjd=0.02
    jdc = 0.01
    f= np.arange(-np.pi,np.pi,jdjd) # fai
    ff = np.arange(jdc+0.0,1+jdc,jdc) #c

    e = overlap_fai_1d.T
    ee =  overlap_c_1d.T

    #3D 不翻转相乘
    testee = ee**2

    testee = testee**0.5

    ee = testee

    c_mean = np.mean(testee[(testee<1) & (testee>0.03)])

    #g=fai
    g,g_ = np.histogram(e,f)
    g[int(np.pi/jdjd)]=(g[int(np.pi/jdjd)+1]+g[int(np.pi/jdjd)-1])/2
    fai,g,g2,rate,h = extend(g,j,jdjd)

    #gg=c
    gg,gg_ = np.histogram(ee,ff)
    emd = EMD()
    imf2 = emd(gg)
    gg2 = np.sum(imf2[0:,:],0)
    gg2[0:10]=0   #

    hh = np.where(gg2 == np.max(gg2))
    cvalue = jdc*np.mean(hh)+jdc+0.00

    print('c_mean:', c_mean ) #这个是全
    print("fai=",fai)
    print("c_EMD=",cvalue)

    ###画图####
    if filename=='':
        filename = run_parameter['filename']
    # draw_c_fai(gg,gg2,g,g2,i,j,np.mean(h),c_mean,cvalue,fai,filename,run_parameter)
    if _cfai_png_output_enabled(opt, run_parameter):
        draw_c_fai(
            gg,gg2,g,g2,i,j,np.mean(h),c_mean,cvalue,fai,filename,
            run_parameter,np.mean(hh),jdc,jdjd
        )

    return c_mean,fai,cvalue


def draw_c_fai(c,c2,fa,fa2,i,j,h,c_mean,c_EMD,fai,filename,run_parameter,hh,jdc,jdjd):
    import matplotlib.pyplot as plt
    '画图保存'
    fig = plt.figure(figsize=(15,3))
    sub1 = plt.subplot(1, 2, 1)  
    
    c_max = np.max(c2)
    # plt.plot()
    start = 10
    plt.plot(c2)
    plt.title(f'{i+1,j}-C-Modulation depth',fontname='Times New Roman',weight ='bold')
    # sub1.text(start+1,  0,"MEAN:",ha='center', va='bottom')
    sub1.text(0.98, 0.98,'MEAN:'+str(c_mean.round(2)),
             horizontalalignment='right',
             verticalalignment='top',
             transform = sub1.transAxes)
    # sub1.text(start+1,  0,str(c_mean.round(2)),ha='center', va='bottom')    
    sub1.text(hh, 0,str((c_EMD).round(2)),ha='center', va='bottom')
    
    #转变X轴下标
    keduc = [20,40,60,80,100]
    # keduc = np.array(keduc)-20
    keduc2 = []
    for ii in range(len(keduc)):
            keduc2.append( np.round((keduc[ii])*jdc,1) )
    sub1.set_xlim((start, 100))
    sub1.set_xticks(keduc)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    sub1.set_xticklabels(keduc2)
    sub1.set_ylim(bottom=0.)  #Y从0开始

    #####################画fai############
    sub2 = plt.subplot(1, 2, 2)  
    plt.plot(fa)
    plt.plot(fa2)
    plt.title(f'{i+1,j}-φ-phase',fontname='Times New Roman',weight ='bold')
    sub2.text(h, 0,fai.round(2),ha='center', va='bottom')
    kedu = [0,157,314,471,628]
    kedu2 = []
    for ii in range(len(kedu)):
        if kedu[ii]<=314:
            pianyi = np.pi
        else:
            pianyi = 3*np.pi

        if kedu[ii]==314:
            kedu2.append( '±'+str(abs(np.round((kedu[ii]*jdjd-pianyi),2))) )
        else:
            kedu2.append( np.round((kedu[ii]*jdjd-pianyi),2) )
    sub2.set_xticks(kedu)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    sub2.set_xticklabels(kedu2)
    sub2.set_ylim(bottom=0.)  #Y从0开始

    cfai_png = plt2np(plt)

    name = _U.save_path(f'{i+1,j}-cfai-{filename}.png',run_parameter,sub_filename='cfai'+'-'+str(run_parameter['pzzz'])+'-'+str(run_parameter['Dr_sub']))
    plt.savefig(name)

    # plt.show() #cmd 模式下 不显示图片
    return cfai_png

def plt2np(plt):
    # 引入 FigureCanvasAgg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # 引入 Image
    import PIL.Image as Image
    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    
    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    
    return rgb_image


def extend(Initial_phase_curve,j,jdjd):
    #如果没在中心，则数据copy一份拼接在右边
    IPC = Initial_phase_curve
    IPC_copy = np.copy(IPC)
    IPC_4pi = np.concatenate((IPC, IPC_copy), axis=0)


    emd = EMD()
    imf = emd(IPC_4pi)

    '按照占比求EMD'
    rate = np.max(IPC_4pi) / np.sum(IPC_copy)
    print( '占比：',rate )
    if rate > 0.02 :              #尖锐
        EMD_curve = np.sum(imf[:,:],0)

    elif rate > 0.013:            #一般
        EMD_curve = np.sum(imf[1:,:],0)

    elif rate > 0.008:            #不太尖锐
        EMD_curve = np.sum(imf[2:,:],0)

    else:# rate > 0.4:           #非常不尖锐
        EMD_curve = np.sum(imf[3:,:],0)

    nx = IPC_4pi.reshape(-1).shape

    '目前IPC_4pi是4pi'
    EMD_curve_pre = np.copy(EMD_curve)
    EMD_curve_pre[int(nx[0]/2):]=0  #后半部分置0
    idx_max = np.where(EMD_curve_pre==np.max(EMD_curve_pre)) #在前半部分找最大值
    fai = np.mean(idx_max)*jdjd - np.pi
    if fai < -2:  #去后半部分拟合
        EMD_curve_later = np.copy(EMD_curve)
        EMD_curve_later[:int(nx[0]/2)]=0  #前半部分置0
        idx_max = np.where(EMD_curve_later==np.max(EMD_curve_later)) #在前半部分找最大值
        fai = np.mean(idx_max)*jdjd - 3*np.pi

        return fai,IPC_4pi,EMD_curve_later,rate,idx_max
    return fai,IPC_4pi,EMD_curve_pre,rate,idx_max

###################################################




























###################################################
def parameter_estimation(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    #常规，先求2阶 再求1阶
    assert len(ck.shape)==4
    #超参数xy
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']          #范围在±20之内
    
    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N =='MP' and opt.Norders==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz_re):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N =='MP' and opt.Norders==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    PGxy = np.zeros((opt.Nangles,opt.Norders,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Norders))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Norders,3)) #
    pgz = 0
    if opt.flag_N=='MP' and opt.Norders==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):# 无解卷积
                RL_output.append(ck[i*opt.Norders+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')
        Ns = opt.Norders

        for j in range(1,opt.Norders ):
            # ±2            
            if j <=1 : #######################
                #整数
                opt.pg = pg

                tempck0=_U.save_path('ck0.npy',run_parameter)
                tempck2=_U.save_path('ck2.npy',run_parameter)
                np.save(tempck0,RL_output[0])
                np.save(tempck2,RL_output[j])
                print('tempck0',tempck0)
                pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 2D，中间层互相关
                print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                #亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                
                #求C fai
                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                
                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2: ##########################
                PGxy[i,j,:] = -PGxy[i,1,:]
                # cfai[i,j,0] = cfai[i,1,0]
                # cfai[i,j,2] = cfai[i,1,2]
                # cfai[i,j,1] = -cfai[i,1,1]
                pgx,pgy = PGxy[i,j,:]

                #求C fai
                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])

            
            #±1
            #情况1 知晓±2 拿来直接用 
            elif j>=3 and opt.Norders == 7  and opt.nz!=1: ###########MP stack##################
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                
                #求pz整数像素
                opt.pg = pg//2 #int(opt.pxy//2)
                pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')

                #求亚像素
                # 自己求pz
                if run_parameter['pzzz']==0:
                     #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                # 给一个亚像素值，但还是要算
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #按照给的理想值 range=1 再微调一下
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
 
                    #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                    #新方法 总和/总和 得到c fai
                    #策略2 
                    # maxz_y,maxx_y,maxy_y = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)  #仅在整数上判断方向
                    
                    #策略3
                    # maxz_y,maxx_y,maxy_y = get_subpz2_2(opt,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)
                    # pgx =  maxx_y-opt.nx
                    # pgy =  maxy_y-opt.ny
                    # pgz =  maxz_y-(opt.nz+2*opt.Zpadding)

                # 固定pz值
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 : #按照给的lai
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                else:
                    raise('参数错误')
                print(f'亚像素pz{j}阶=',pgz,'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 


                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz


            elif j>=3 and opt.Norders == 7  and opt.nz==1: ###########MP slice##################
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                pgz = 0

                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz
                
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=[]
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2) )
        for j in range(3,opt.Norders):   #±1
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2)
    plong = np.mean(np.array(plong))
    

    if Debug==True: #4
        lihe2d = np.array(opt.lihe2d)
        overlap_mask3d = np.array(opt.overlap_mask3d) 
        abscm3d = np.array(opt.abscm3d)
        angcm3d = np.array(opt.angcm3d)
        
        kzz,kxx,kyy = lihe2d.shape
        lihe2d[:,kxx//2,kyy//2] = 0.1
        lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) #加一个投影
        lihe2d = np.insert(lihe2d,0,values = np.min(lihe2d,axis=0),axis=0) #加一个投影

        _U.save_tif_v21(f'lihe_±2-[{filename}].tif',lihe2d,run_parameter,sub_filename='P')
        # _U.save_tif_v21(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_),run_parameter)
        # _U.save_tif_v21(f'overlap3d_c_±1-[{filename}].tif',abscm3d,run_parameter)
        # _U.save_tif_v21(f'overlap3d_fai_±1-[{filename}].tif',angcm3d,run_parameter)


        #     lihe3d = np.array(opt.lihe3d)
        #     lihe3dxy = np.array(opt.lihe3dxy)
        #     # _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
        #     # _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
        #     # _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
        #     # _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
        #     _U.save_tif_v21(f'overlap3d_c_±1-[{filename}].tif',abscm3d,run_parameter)
        #     _U.save_tif_v21(f'overlap3d_fai_±1-[{filename}].tif',angcm3d,run_parameter)
        #     _U.save_tif_v21(f'lihe_±1-[{filename}].tif',lihe3d,run_parameter)
        #     _U.save_tif_v21(f'lihe_±1xy-[{filename}].tif',lihe3dxy,run_parameter)



    # del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz



def parameter_estimation1_2(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    #常规，先求2阶 再求1阶
    #对1阶来说，为了频谱对称，只求一阶的3,4   然后 3的给6 4的给5
    # p = -p  c = c , fai=-fai
    assert len(ck.shape)==4
    #超参数xy
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']          #范围在±20之内
    
    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N =='MP' and opt.Norders==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz_re):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N =='MP' and opt.Norders==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    PGxy = np.zeros((opt.Nangles,opt.Norders,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Norders))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Norders,3)) #
    pgz = 0
    if opt.flag_N=='MP' and opt.Norders==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):# 无解卷积
                RL_output.append(ck[i*opt.Norders+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        print(f'【第{i+1}个方向】:')
        Ns = opt.Norders

        for j in range(1,opt.Norders ):
            # ±2            
            if j <=1 : #######################
                #整数
                opt.pg = pg

                tempck0=_U.save_path('ck0.npy',run_parameter)
                tempck2=_U.save_path('ck2.npy',run_parameter)
                np.save(tempck0,RL_output[0])
                np.save(tempck2,RL_output[j])
                print('tempck0',tempck0)
                pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 2D，中间层互相关
                print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                #亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                
                #求C fai
                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                
                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2: ##########################
                #p = -p  c = c , fai=-fai
                PGxy[i,j,:] = -PGxy[i,1,:]
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]

            #±1
            #情况1 知晓±2 拿来直接用 
            elif j>=3 and j<=4 and opt.Norders == 7  and opt.nz!=1: ###########MP stack##################
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                
                #求pz整数像素
                opt.pg = pg//2 #int(opt.pxy//2)
                pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')

                #求亚像素
                # 自己求pz
                if run_parameter['pzzz']==0:
                     #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                # 给一个亚像素值，但还是要算
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #按照给的理想值 range=1 再微调一下
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
 
                    #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                    #新方法 总和/总和 得到c fai
                    #策略2 
                    # maxz_y,maxx_y,maxy_y = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)  #仅在整数上判断方向
                    
                    #策略3
                    # maxz_y,maxx_y,maxy_y = get_subpz2_2(opt,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)
                    # pgx =  maxx_y-opt.nx
                    # pgy =  maxy_y-opt.ny
                    # pgz =  maxz_y-(opt.nz+2*opt.Zpadding)

                # 固定pz值
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 : #按照给的lai
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                else:
                    raise('参数错误')
                print(f'亚像素pz{j}阶=',pgz,'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 


                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz



            elif j>=3 and j<=4 and opt.Norders == 7  and opt.nz==1: ###########MP slice##################
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                pgz = 0

                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz

            elif j >= 5 and opt.Norders == 7:
                #p = -p  c = c , fai=-fai
                if j == 5:
                    PGxy[i,j,:] = -PGxy[i,4,:]

                    cfai[i,j,0] = cfai[i,4,0]
                    cfai[i,j,2] = cfai[i,4,2]
                    cfai[i,j,1] = -cfai[i,4,1]
                    PGz[i,j] =  PGz[i,3]      #warn  pz = 5 match 3
                elif j==6:
                    PGxy[i,j,:] = -PGxy[i,3,:]
                    cfai[i,j,0] = cfai[i,3,0]
                    cfai[i,j,2] = cfai[i,3,2]
                    cfai[i,j,1] = -cfai[i,3,1]
                    PGz[i,j] =  PGz[i,4]      #warn pz =6 match 4
                else:
                    raise('j error')

            elif j==3 and opt.Norders == 5:
                PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                pgz = 0

                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                # PGz[i,j] = pgz        #得到精确的Pz
            elif j==4 and  opt.Norders == 5:
                PGxy[i,j,:] = -PGxy[i,3,:]

                cfai[i,j,0] = cfai[i,3,0]
                cfai[i,j,2] = cfai[i,3,2]
                cfai[i,j,1] = -cfai[i,3,1]
                # PGz[i,j] =  PGz[i,3]      #warn  pz = 5 match 3
            

                
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=[]
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2) )
        for j in range(3,opt.Norders):   #±1
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2)
    plong = np.mean(np.array(plong))
    

    if Debug==True: #4
        Dr_sub = run_parameter['Dr_sub']
        lihe2d = np.array(opt.lihe2d)
        overlap_mask3d = np.array(opt.overlap_mask3d) 
        abscm3d = np.array(opt.abscm3d)
        angcm3d = np.array(opt.angcm3d)
        
        kzz,kxx,kyy = lihe2d.shape
        lihe2d[:,kxx//2,kyy//2] = 0.1
        lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) #加一个投影
        lihe2d = np.insert(lihe2d,0,values = np.min(lihe2d,axis=0),axis=0) #加一个投影

        _U.save_tif_v21(f'lihe_±2-[{Dr_sub}-{filename}].tif',lihe2d,run_parameter,sub_filename='P')
        
        # _U.save_tif_v21(f'overlap3d_±1-[{filename}].tif',overlap_mask3d[[0,1,2]].astype(np.bool_),run_parameter)
        # _U.save_tif_v21(f'overlap3d_c_±1-[{filename}].tif',abscm3d[[0,1,2]],run_parameter)
        # _U.save_tif_v21(f'overlap3d_fai_±1-[{filename}].tif',angcm3d[[0,1,2]],run_parameter)

    # del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz



def parameter_estimation2(opt,ck,PGxy,PGz,filename='',RL=False,Debug=False,run_parameter=''):
    #传入PGxy，计算cfai而已
    assert len(ck.shape)==4
    #超参数xy
    opt.pg = int(opt.pxy)         #pg这里是2阶的pxy=r/2
    opt.fanwei = 10#50            #范围在±20之内
    
    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N =='MP' and opt.Norders==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N =='MP' and opt.Norders==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    # PGxy = np.zeros((opt.Nangles,opt.Norders,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    # PGz = np.zeros((opt.Nangles,opt.Norders))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Norders,3)) #
    pgz = 0
    if opt.flag_N =='MP' and opt.Norders==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):# 无解卷积
                RL_output.append(ck[i*opt.Norders+j])


            ########################################
            # print('参数估计时，使用空域投影，2D上求')
            # RL_output=[]
            # for j in range(opt.Norders):
            #     tempr = _U.ifftn(ck[i*opt.Norders+j])
            #     tempr_2D = np.mean(tempr,axis=0)
            #     tempk_2D = _U.fftn(tempr_2D)
            #     tempk_3D = _U.image_reshape(tempk_2D,dim=3)
            #     # _U.save_tif('tempk_3D.tif',abs(tempk_3D))
            #     RL_output.append(tempk_3D)
            ########################################
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        

        print(f'【第{i+1}个方向】:')

        Ns = opt.Norders
        # for j in range(1,opt.Norders):
        for j in range(1,opt.Norders ):
            pgx = PGxy[i,j,0]
            pgy = PGxy[i,j,1]          
            if j ==1 :  #±2阶求P
                
                print('2jie',pgx,pgy)
                #已知亚像素坐标 求C fai

                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai,c_EMD)

                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2: ##########################
                #p = -p  c = c , fai=-fai
                
                # PGxy[i,j,:] = -PGxy[i,1,:] #不要修改PGxy
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]


            elif j>=3 and j<=4 and opt.flag_N =='MP' and opt.Norders==7:      # -1，-1。 pgx，pgy用2阶的， pz这里尝试 取横向一般位置 竖着互相关，得到整数pz

                print('MP 1jie')
                pgz = PGz[i,j]

                print(f'亚像素pz{j}阶=',PGz[i,j],'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 
                

                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)

                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j >= 5 and opt.Norders == 7:
                #p = -p  c = c , fai=-fai
                if j == 5:
                    # PGxy[i,j,:] = -PGxy[i,4,:]

                    cfai[i,j,0] = cfai[i,4,0]
                    cfai[i,j,2] = cfai[i,4,2]
                    cfai[i,j,1] = -cfai[i,4,1]
                    PGz[i,j] =  PGz[i,3]      #warn  pz = 5 match 3
                elif j==6:
                    # PGxy[i,j,:] = -PGxy[i,3,:]
                    cfai[i,j,0] = cfai[i,3,0]
                    cfai[i,j,2] = cfai[i,3,2]
                    cfai[i,j,1] = -cfai[i,3,1]
                    PGz[i,j] =  PGz[i,4]      #warn pz =6 match 4
                else:
                    raise('j error')



    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime )
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Norders):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)


    return cfai,PGxy,PGz,plong,pgz



def parameter_estimation3(opt,ck,PGxy,filename='',RL=False,Debug=False,run_parameter='',jieshu = [1,2,3,4,5,6]):
    #传入PGxy，计算PGz,cfai — GPU entry with device policy / auto fallback
    rp = run_parameter if isinstance(run_parameter, dict) else {}
    def _work():
        return _parameter_estimation3_impl(
            opt, ck, PGxy, filename=filename, RL=RL, Debug=Debug,
            run_parameter=run_parameter, jieshu=jieshu
        )
    return _run_with_device_fallback(opt, rp, _work, tag='parameter_estimation3')


def _parameter_estimation3_impl(opt,ck,PGxy,filename='',RL=False,Debug=False,run_parameter='',jieshu = [1,2,3,4,5,6]):
    #传入PGxy，计算PGz,cfai
    #jieshu 控制 参数计算哪几阶（注意 只有1 2 计算PGxy 和cfai ；3456用1 2的PGxy，计算cfai）
    assert len(ck.shape)==4
    #超参数xy
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']          #范围在±20之内

    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数

    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N =='MP' and opt.Norders==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz_re):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N =='MP' and opt.Norders==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    # PGxy = np.zeros((opt.Nangles,opt.Norders,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Norders))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Norders,3)) #
    pgz = 0
    if opt.flag_N =='MP' and opt.Norders==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):# 无解卷积
                RL_output.append(ck[i*opt.Norders+j])


            ########################################
            # print('参数估计时，使用空域投影，2D上求')
            # RL_output=[]
            # for j in range(opt.Norders):
            #     tempr = _U.ifftn(ck[i*opt.Norders+j])
            #     tempr_2D = np.mean(tempr,axis=0)
            #     tempk_2D = _U.fftn(tempr_2D)
            #     tempk_3D = _U.image_reshape(tempk_2D,dim=3)
            #     # _U.save_tif('tempk_3D.tif',abs(tempk_3D))
            #     RL_output.append(tempk_3D)
            ########################################
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        

        print(f'【第{i+1}个方向】:')

        Ns = opt.Norders
        
        # jiesu = 
        for j in jieshu:
            pgx = PGxy[i,j,0]
            pgy = PGxy[i,j,1]          
            if j <=1 :  #±2阶
                
                print('2jie',pgx,pgy)
                #已知亚像素坐标 求C fai

                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai,c_EMD)

                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2: ##########################
                #p = -p  c = c , fai=-fai
                
                # PGxy[i,j,:] = -PGxy[i,1,:]
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]

            elif j>=3 and j<=4 and opt.flag_N =='MP' and opt.Norders==7:      # -1，-1。 pgx，pgy用2阶的， pz这里尝试 取横向一般位置 竖着互相关，得到整数pz

                # print('MP 1jie')
                # 自己求pz
                if run_parameter['pzzz']==0:
                    #求pz整数像素
                    opt.pg = pg//2 #int(opt.pxy//2)
                    pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                    print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')
                    
                     #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                # 给一个亚像素值，但还是要算
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #按照给的理想值 range=1 再微调一下
                    #求pz整数像素
                    opt.pg = pg//2 #int(opt.pxy//2)
                    pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                    print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')
                    
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
 
                    #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                    #新方法 总和/总和 得到c fai
                    #策略2 
                    # maxz_y,maxx_y,maxy_y = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)  #仅在整数上判断方向
                    
                    #策略3
                    # maxz_y,maxx_y,maxy_y = get_subpz2_2(opt,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)
                    # pgx =  maxx_y-opt.nx
                    # pgy =  maxy_y-opt.ny
                    # pgz =  maxz_y-(opt.nz+2*opt.Zpadding)

                # 固定pz值
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 : #按照给的lai
                    print('直接给PZ')
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                else:
                    raise('参数错误')

                print(f'亚像素pz{j}阶=',pgz,'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 
                

                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)

                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGz[i,j] = pgz        #得到精确的Pz
            
            elif j >= 5 and opt.Norders == 7:
                #p = -p  c = c , fai=-fai
                if j == 5:
                    # PGxy[i,j,:] = -PGxy[i,4,:]

                    cfai[i,j,0] = cfai[i,4,0]
                    cfai[i,j,2] = cfai[i,4,2]
                    cfai[i,j,1] = -cfai[i,4,1]
                    PGz[i,j] =  PGz[i,3]      #warn  pz = 5 match 3
                elif j==6:
                    # PGxy[i,j,:] = -PGxy[i,3,:]
                    cfai[i,j,0] = cfai[i,3,0]
                    cfai[i,j,2] = cfai[i,3,2]
                    cfai[i,j,1] = -cfai[i,3,1]
                    PGz[i,j] =  PGz[i,4]      #warn pz =6 match 4
                else:
                    raise('j error')
            
            elif j==3 and opt.flag_N =='MP' and opt.Norders==5:             #MP slice重构
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)

                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGz[i,j] = pgz        #得到精确的Pz
            elif j==4 and opt.flag_N =='MP' and opt.Norders==5:
                cfai[i,j,0] = cfai[i,3,0]
                cfai[i,j,2] = cfai[i,3,2]
                cfai[i,j,1] = -cfai[i,3,1]
                # PGz[i,j] =  PGz[i,3]      #warn  pz = 5 match 3

    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime )
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Norders):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)

    if Debug==True: #4
        Dr_sub = run_parameter['Dr_sub']
        overlap_mask3d = np.array(opt.overlap_mask3d) 
        abscm3d = np.array(opt.abscm3d)
        angcm3d = np.array(opt.angcm3d)
        

        # _U.save_tif_v21(f'overlap3d_±1-[{filename}].tif',overlap_mask3d[[0,1,2]].astype(np.bool_),run_parameter)
        # _U.save_tif_v21(f'overlap3d_c_±1-[{filename}].tif',abscm3d[[0,1,2]],run_parameter)
        # _U.save_tif_v21(f'overlap3d_fai_±1-[{filename}].tif',angcm3d[[0,1,2]],run_parameter)

    # del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d


    return cfai,PGxy,PGz,plong,pgz


def parameter_estimation_1(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    #常规- 先求1阶给2阶，bio数据上2阶互相关没有亮点
    assert len(ck.shape)==4
    #超参数xy
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']          #范围在±20之内
    
    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N =='MP' and opt.Norders==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N =='MP' and opt.Norders==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    PGxy = np.zeros((opt.Nangles,opt.Norders,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Norders))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Norders,3)) #
    pgz = 0
    if opt.flag_N =='MP' and opt.Norders==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Norders):# 无解卷积
                RL_output.append(ck[i*opt.Norders+j])
            print('参数估计时，使用空域投影，2D上求')
            RL_output=[]
            for j in range(opt.Norders):
                tempr = _U.ifftn(ck[i*opt.Norders+j])
                tempr_2D = np.mean(tempr,axis=0)
                tempk_2D = _U.fftn(tempr_2D)
                tempk_3D = _U.image_reshape(tempk_2D,dim=3)
                # _U.save_tif('tempk_3D.tif',abs(tempk_3D))
                RL_output.append(tempk_3D)
        
        RL_output = np.array(RL_output)
        _U.save_tif('RL_output.tif',abs(RL_output))
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')
        Ns = opt.Norders

        # for j in range(1,opt.Norders ):
        for j in [3,4,5,6,1,2]:
            # ±2            
            if j <=2 : #######################
                #######################------2阶计算--------
                #整数
                # opt.pg = pg

                # pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 2D，中间层互相关
                # print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                # #亚像素
                # pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                # print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                
                ########################-------2阶赋值
                pgx = PGxy[i,(j+1)*2,0]   # 4/6
                pgy = PGxy[i,(j+1)*2,1]


                print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')

                ########################

                #求C fai
                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                
                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])
          

            #±1
            #情况1 知晓±2 拿来直接用 
            elif j>=3 and opt.Norders == 7  and opt.nz!=1: ###########MP stack##################
                ##################根据1阶计算整数##########
                #求pz整数像素
                opt.pg = pg//2 #int(opt.pxy//2)
                pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')
                
                # pxint,pyint = get_intpxy(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                # print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')
                
                #求pxy亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                

                # #1阶给2阶
                # PGxy[i,int(np.ceil(j/2)-1),:] = 2*np.array([pgx,pgy])  #给2阶整数倍的p/2

                #求亚像素
                # 自己求pz
                if run_parameter['pzzz']==0:
                     #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                # 给一个亚像素值，但还是要算
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #按照给的理想值 range=1 再微调一下
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
 
                    #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                    #新方法 总和/总和 得到c fai
                    #策略2 
                    # maxz_y,maxx_y,maxy_y = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)  #仅在整数上判断方向
                    
                    #策略3
                    # maxz_y,maxx_y,maxy_y = get_subpz2_2(opt,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)
                    # pgx =  maxx_y-opt.nx
                    # pgy =  maxy_y-opt.ny
                    # pgz =  maxz_y-(opt.nz+2*opt.Zpadding)

                # 固定pz值
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 : #按照给的lai
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                else:
                    raise('参数错误')
                # print(f'亚像素pz{j}阶=',pgz,'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 


                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz


            elif j>=3 and opt.Norders == 7  and opt.nz==1: ###########MP slice##################
                ##################根据1阶计算整数##########
                opt.pg = pg//2 #int(opt.pxy//2)

                pxint,pyint = get_intpxy(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                # print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')
                
                #求pxy亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                
                # #给2阶  3,4->1   5、6->2
                # PGxy[i,int(np.ceil(j/2)-1),:] = 2*np.array([pgx,pgy])  #给2阶整数倍的p/2
                
                pgz=0
                #求CFAI
                c,fai,c_EMD = cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz
                
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=[]
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2) )
        for j in range(3,opt.Norders):   #±1
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2)
    plong = np.mean(np.array(plong))
    

    print(PGxy)
    print(PGz)
    print(cfai)

    # if Debug==True: #4

    #     overlap_mask2d = np.array(opt.overlap_mask2d) 
    #     abscm2d = np.array(opt.abscm2d)
    #     angcm2d = np.array(opt.angcm2d)
    #     lihe2d = np.array(opt.lihe2d)
        
    #     kzz,kxx,kyy = lihe2d.shape
    #     lihe2d[:,kxx//2,kyy//2] = 0.1
    #     lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) #加一个投影
    #     lihe2d = np.insert(lihe2d,0,values = np.min(lihe2d,axis=0),axis=0) #加一个投影
    #     _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
    #     _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
    #     _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        
    #     _U.save_tif(f'lihe_±2-[{filename}].tif',lihe2d)

    #     if opt.Norders==7:
    #         overlap_mask3d = np.array(opt.overlap_mask3d) 
    #         abscm3d = np.array(opt.abscm3d)
    #         angcm3d = np.array(opt.angcm3d)
    #         lihe3d = np.array(opt.lihe3d)
    #         lihe3dxy = np.array(opt.lihe3dxy)
    #         _U.save_tif(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_))
    #         _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
    #         _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
    #         _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
    #         _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
    # del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    # del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz



def parameter_estimationP2(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    # 仅计算2阶P,ck(3,5) — GPU entry with device policy / auto fallback
    rp = run_parameter if isinstance(run_parameter, dict) else {}
    def _work():
        resolved_device = getattr(opt, 'device', torch.device('cpu'))
        if torch.device(resolved_device).type == 'cuda':
            _log_pe('parameter_estimationP2: using CPU for deterministic PGxy search')
            opt.device = torch.device('cpu')
        try:
            return _parameter_estimationP2_impl(
                opt, ck, filename=filename, RL=RL, Debug=Debug,
                run_parameter=run_parameter
            )
        finally:
            opt.device = resolved_device
    return _run_with_device_fallback(opt, rp, _work, tag='parameter_estimationP2')


def _parameter_estimationP2_impl(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    # 仅计算2阶P,ck(3,5)
    assert len(ck.shape)==4
    Norders = 5
    opt._lihe_scalar_cache = {}
    opt._lihe_scalar_cache_hits = 0
    opt._lihe_scalar_cache_misses = 0
    
    PGxy = np.zeros((opt.Nangles,Norders,2))
    #超参数xy
    if run_parameter['pg']!=0:
        pg = run_parameter['pg']
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        pg = int(opt.pxy)
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']          #范围在±20之内
    
    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(Norders):# 无解卷积
                RL_output.append(ck[i*Norders+j])


        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4        

        print(f'【第{i+1}个方向】:')
        # for j in range(1,opt.Norders):
        for j in [1,2]:        
            if j ==1 :  #±2阶求P
                opt.pg = pg
                pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 2D，中间层互相关
                print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                #亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                
                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                
            elif j == 2: ##########################
                #p = -p
                PGxy[i,j,:] = -PGxy[i,1,:]
            

    lihe2d = np.array(opt.lihe2d)
    kzz,kxx,kyy = lihe2d.shape
    lihe2d[:,kxx//2,kyy//2] = 0.1
    lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0)
    Dr_sub = run_parameter['Dr_sub']
    _U.save_tif_v21(
        f'lihe_±2-[{filename}]-{Dr_sub}.tif',
        lihe2d,
        run_parameter,
        sub_filename='P',
    )
    _log_pe(
        'PGxy scalar cache: '
        f'hits={opt._lihe_scalar_cache_hits}, '
        f'misses={opt._lihe_scalar_cache_misses}'
    )
    del opt._lihe_scalar_cache
    return PGxy



def parameter_estimation_02(opt,ck,PGxy,Norders,filename='',RL=False,Debug=False,run_parameter=''):
    # 55 传入PGxy，计算2阶cfai — GPU entry with device policy / auto fallback
    rp = run_parameter if isinstance(run_parameter, dict) else {}
    def _work():
        return _parameter_estimation_02_impl(
            opt, ck, PGxy, Norders, filename=filename, RL=RL, Debug=Debug, run_parameter=run_parameter
        )
    return _run_with_device_fallback(opt, rp, _work, tag='parameter_estimation_02')


def _parameter_estimation_02_impl(opt,ck,PGxy,Norders,filename='',RL=False,Debug=False,run_parameter=''):
     # 55 传入PGxy，计算2阶cfai
     # 返回cfai (3,3,2)
    assert len(ck.shape)==4
    
    cfai = np.zeros((opt.Nangles,3,3)) #
    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(Norders):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*Norders+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(Norders):# 无解卷积
                RL_output.append(ck[i*Norders+j])


        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        

        print(f'【第{i+1}个方向】:')
        for j in range(1,Norders):
            pgx = PGxy[i,j,0]
            pgy = PGxy[i,j,1]          
            if j <=1 :  #±2阶求P
                
                print('2jie',pgx,pgy)
                #已知亚像素坐标 求C fai

                c,fai,c_EMD= cal_c_fai(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),filename=filename,run_parameter=run_parameter) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai,c_EMD)

                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2: ##########################
                #p = -p  c = c , fai=-fai
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]
                
    return cfai




