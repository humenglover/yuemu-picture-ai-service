#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控工具 - 自愈监控
用于监控服务器 CPU、内存、磁盘、GPU 等资源状态
"""

import psutil
import platform
from typing import Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger


def get_gpu_info():
    """
    获取 GPU 信息（如果可用）
    支持 NVIDIA GPU（通过 pynvml）
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            gpu_info.append({
                "index": i,
                "name": name.decode('utf-8') if isinstance(name, bytes) else name,
                "memory_total_gb": round(memory_info.total / 1024**3, 2),
                "memory_used_gb": round(memory_info.used / 1024**3, 2),
                "memory_free_gb": round(memory_info.free / 1024**3, 2),
                "memory_percent": round(memory_info.used / memory_info.total * 100, 1),
                "gpu_utilization": utilization.gpu,
                "memory_utilization": utilization.memory,
                "temperature": temperature
            })
        
        pynvml.nvmlShutdown()
        return gpu_info
    except ImportError:
        knowledge_logger.debug('[SYSTEM_MONITOR] pynvml 未安装，无法获取 GPU 信息')
        return None
    except Exception as e:
        knowledge_logger.debug(f'[SYSTEM_MONITOR] 获取 GPU 信息失败: {str(e)}')
        return None


def get_system_status():
    """
    获取系统完整状态信息
    """
    try:
        # CPU 信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息（Windows 使用 C:\ 而不是 /）
        try:
            disk = psutil.disk_usage('/')
        except:
            try:
                disk = psutil.disk_usage('C:\\')
            except:
                disk = psutil.disk_usage('C:/')
        
        # 网络信息
        net_io = psutil.net_io_counters()
        
        # 进程信息
        process_count = len(psutil.pids())
        
        # GPU 信息
        gpu_info = get_gpu_info()
        
        # 系统信息
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version()
        }
        
        status = {
            "cpu": {
                "usage_percent": round(cpu_percent, 1),
                "core_count": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 0) if cpu_freq and hasattr(cpu_freq, 'current') and cpu_freq.current else None,
                "status": "正常" if cpu_percent < 70 else "繁忙" if cpu_percent < 90 else "高负载"
            },
            "memory": {
                "total_gb": round(memory.total / 1024**3, 2),
                "used_gb": round(memory.used / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "usage_percent": round(memory.percent, 1),
                "status": "充足" if memory.percent < 70 else "紧张" if memory.percent < 85 else "告急"
            },
            "disk": {
                "total_gb": round(disk.total / 1024**3, 2),
                "used_gb": round(disk.used / 1024**3, 2),
                "free_gb": round(disk.free / 1024**3, 2),
                "usage_percent": round(disk.percent, 1),
                "status": "充足" if disk.percent < 70 else "紧张" if disk.percent < 85 else "告急"
            },
            "network": {
                "bytes_sent_gb": round(net_io.bytes_sent / 1024**3, 2),
                "bytes_recv_gb": round(net_io.bytes_recv / 1024**3, 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "process": {
                "count": process_count
            },
            "system": system_info
        }
        
        # 添加 GPU 信息（如果可用）
        if gpu_info:
            status["gpu"] = gpu_info
        
        return status
        
    except Exception as e:
        knowledge_logger.error(f'[SYSTEM_MONITOR] 获取系统状态失败: {str(e)}')
        import traceback
        knowledge_logger.error(f'[SYSTEM_MONITOR] 堆栈: {traceback.format_exc()}')
        return {"error": str(e)}


def generate_friendly_report(status: dict) -> str:
    """
    生成拟人化的系统状态报告
    """
    if "error" in status:
        return f"抱歉，无法获取系统状态：{status['error']}"
    
    try:
        cpu = status.get("cpu", {})
        memory = status.get("memory", {})
        disk = status.get("disk", {})
        gpu_list = status.get("gpu", [])
        
        # 构建拟人化报告
        report_parts = []
        
        # 获取数值，确保类型安全
        cpu_usage = float(cpu.get("usage_percent", 0))
        mem_usage = float(memory.get("usage_percent", 0))
        
        # 开场白
        if cpu_usage < 50 and mem_usage < 60:
            report_parts.append("系统健康诊断：服务器当前运行状态正常，各项指标均在控制范围内。")
        elif cpu_usage < 80 and mem_usage < 80:
            report_parts.append("系统健康诊断：服务器正处于中等负载工作状态，响应平稳。")
        else:
            report_parts.append("系统健康诊断：服务器目前负载较高，系统正在调度资源以维持稳定性。")
        
        # CPU 状态
        cpu_cores = int(cpu.get("core_count", 0))
        if cpu_usage < 50:
            report_parts.append(f"\n- **处理器 (CPU)**: {cpu_cores} 核心，使用率 {cpu_usage:.1f}%（资源冗余充足）")
        elif cpu_usage < 80:
            report_parts.append(f"\n- **处理器 (CPU)**: {cpu_cores} 核心，使用率 {cpu_usage:.1f}%（运算任务分配中）")
        else:
            report_parts.append(f"\n- **处理器 (CPU)**: {cpu_cores} 核心，使用率 {cpu_usage:.1f}%（高负载运转中）")
        
        # 内存状态
        mem_available = float(memory.get("available_gb", 0))
        mem_total = float(memory.get("total_gb", 0))
        if mem_usage < 60:
            report_parts.append(f"\n- **物理内存**: {mem_available:.1f}GB 可用（总共 {mem_total:.1f}GB），寻址空间充足")
        elif mem_usage < 85:
            report_parts.append(f"\n- **物理内存**: {mem_available:.1f}GB 可用（总共 {mem_total:.1f}GB），使用率 {mem_usage:.1f}%")
        else:
            report_parts.append(f"\n- **物理内存**: {mem_available:.1f}GB 可用（总共 {mem_total:.1f}GB），使用率 {mem_usage:.1f}%（可用空间紧张）")
        
        # GPU 状态（如果有）
        if gpu_list:
            for gpu in gpu_list:
                gpu_name = str(gpu.get("name", "Unknown GPU"))
                gpu_mem_percent = float(gpu.get("memory_percent", 0))
                gpu_util = float(gpu.get("gpu_utilization", 0))
                gpu_temp = int(gpu.get("temperature", 0))
                
                if gpu_mem_percent < 60 and gpu_util < 60:
                    report_parts.append(f"\n- **图形处理器 (GPU)** ({gpu_name}): 显存占用 {gpu_mem_percent:.1f}%，算力使用 {gpu_util}%，核心温度 {gpu_temp}°C")
                elif gpu_mem_percent < 85 and gpu_util < 85:
                    report_parts.append(f"\n- **图形处理器 (GPU)** ({gpu_name}): 显存占用 {gpu_mem_percent:.1f}%，算力使用 {gpu_util}%，核心温度 {gpu_temp}°C（执行并行计算中）")
                else:
                    report_parts.append(f"\n- **图形处理器 (GPU)** ({gpu_name}): 显存占用 {gpu_mem_percent:.1f}%，算力使用 {gpu_util}%，核心温度 {gpu_temp}°C（并发处理队列拥堵）")
        
        # 磁盘状态
        disk_usage = float(disk.get("usage_percent", 0))
        disk_free = float(disk.get("free_gb", 0))
        if disk_usage < 70:
            report_parts.append(f"\n- **存储磁盘**: {disk_free:.1f}GB 可用空间，存储余量充足")
        elif disk_usage < 85:
            report_parts.append(f"\n- **存储磁盘**: {disk_free:.1f}GB 可用空间，建议定期整理多余文件")
        else:
            report_parts.append(f"\n- **存储磁盘**: {disk_free:.1f}GB 可用空间（存储空间紧张）")
        
        # 总结建议
        if cpu_usage > 85 or mem_usage > 85 or (gpu_list and any(float(g.get("memory_percent", 0)) > 85 for g in gpu_list)):
            report_parts.append("\n\n**运维建议**: 系统当前面临高并发负载，系统正在全力调度集群算力资源，建议稍后重新下发重度计算任务。")
        else:
            report_parts.append("\n\n**运行状态**: 系统当前指标平稳，随时可以响应高并发请求。")
        
        return "".join(report_parts)
    
    except Exception as e:
        knowledge_logger.error(f'[SYSTEM_MONITOR] 生成报告失败: {str(e)}')
        import traceback
        knowledge_logger.error(f'[SYSTEM_MONITOR] 堆栈: {traceback.format_exc()}')
        return f"生成系统状态报告时出错：{str(e)}"


@tool(description="""查询服务器系统状态（自愈监控工具）。

功能说明：
1. 实时监控服务器 CPU、内存、磁盘、GPU 等资源状态
2. 以拟人化的方式报告系统健康状况
3. 帮助管理员了解服务器负载情况

使用场景：
- 用户抱怨处理速度慢时，查询服务器是否繁忙
- 管理员询问"服务器累吗？"、"系统状态如何？"
- 需要了解当前资源使用情况
- 排查性能问题

参数说明：
- detail_level (可选): 详细程度，可选值：
  * "simple" - 简洁报告（默认，拟人化描述）
  * "detailed" - 详细数据（包含所有技术指标）

返回格式：
- simple 模式：拟人化的状态描述，如"当前内存占用 85%，图像处理队列稍有拥堵，正在全力调度中"
- detailed 模式：JSON 格式的完整系统指标

示例对话：
- 用户："为什么处理这么慢？"
- AI 调用工具后回复："当前内存占用 85%，CPU 使用率 78%，图像处理队列稍有拥堵，正在全力调度中。预计稍等片刻即可完成。"
""")
def check_system_status(detail_level: Optional[str] = "simple") -> str:
    """
    查询服务器系统状态
    
    Args:
        detail_level: 详细程度，"simple" 或 "detailed"
    
    Returns:
        系统状态报告（文本或 JSON）
    """
    import json
    
    try:
        knowledge_logger.info(f'[SYSTEM_MONITOR] 开始查询系统状态 | 详细程度: {detail_level}')
        
        # 获取系统状态
        status = get_system_status()
        
        if "error" in status:
            error_msg = f"无法获取系统状态：{status['error']}"
            knowledge_logger.error(f'[SYSTEM_MONITOR] {error_msg}')
            return error_msg
        
        # 根据详细程度返回不同格式
        if detail_level == "detailed":
            knowledge_logger.info('[SYSTEM_MONITOR] 返回详细数据')
            return json.dumps(status, ensure_ascii=False, indent=2)
        else:
            # 生成拟人化报告
            report = generate_friendly_report(status)
            knowledge_logger.info(f'[SYSTEM_MONITOR] 系统状态查询成功 | CPU: {status["cpu"]["usage_percent"]}% | 内存: {status["memory"]["usage_percent"]}%')
            return report
        
    except Exception as e:
        error_msg = f"查询系统状态时发生错误: {str(e)}"
        knowledge_logger.error(f'[SYSTEM_MONITOR] {error_msg}')
        import traceback
        knowledge_logger.error(f'[SYSTEM_MONITOR] 堆栈: {traceback.format_exc()}')
        return error_msg


@tool(description="""获取系统进程信息。

功能说明：
1. 查看当前运行的进程列表
2. 按 CPU 或内存使用率排序
3. 帮助定位资源占用高的进程

使用场景：
- 排查哪个进程占用资源过高
- 了解系统当前运行的主要任务
- 性能调优和问题诊断

参数说明：
- sort_by (可选): 排序方式，可选值：
  * "cpu" - 按 CPU 使用率排序（默认）
  * "memory" - 按内存使用率排序
- limit (可选): 返回进程数量，默认 10

返回格式：
返回 JSON 格式的进程列表，包含：
- pid: 进程 ID
- name: 进程名称
- cpu_percent: CPU 使用率
- memory_percent: 内存使用率
- status: 进程状态
""")
def get_top_processes(sort_by: Optional[str] = "cpu", limit: Optional[int] = 10) -> str:
    """
    获取系统进程信息
    
    Args:
        sort_by: 排序方式，"cpu" 或 "memory"
        limit: 返回进程数量
    
    Returns:
        JSON 格式的进程列表
    """
    import json
    
    try:
        knowledge_logger.info(f'[SYSTEM_MONITOR] 获取进程信息 | 排序: {sort_by} | 数量: {limit}')
        
        # 获取所有进程
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                processes.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "cpu_percent": round(pinfo['cpu_percent'], 2),
                    "memory_percent": round(pinfo['memory_percent'], 2),
                    "status": pinfo['status']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # 排序
        if sort_by == "memory":
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        else:
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        # 限制数量
        top_processes = processes[:limit]
        
        knowledge_logger.info(f'[SYSTEM_MONITOR] 获取进程信息成功 | 总进程数: {len(processes)} | 返回: {len(top_processes)}')
        
        return json.dumps({
            "total_processes": len(processes),
            "top_processes": top_processes,
            "sort_by": sort_by,
            "message": f"按 {sort_by} 使用率排序的前 {len(top_processes)} 个进程"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"获取进程信息时发生错误: {str(e)}"
        knowledge_logger.error(f'[SYSTEM_MONITOR] {error_msg}')
        return json.dumps({"error": error_msg}, ensure_ascii=False)
