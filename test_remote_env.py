
import ray
from verl.trainer.gui_agent_remote import RemoteDesktopEnv

# Initialize Ray (required for remote actors)
ray.init(ignore_reinit_error=True)

print("=" * 60)
print("Testing RemoteDesktopEnv connection...")
print("=" * 60)

try:
    # 直接测试 RemoteDesktopEnv（不需要 model config）
    env = RemoteDesktopEnv(
        base_url="http://120.255.0.146",
        manager_port=9001,
        vm_name="Ubuntu.qcow2",
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        connect_max_try=3,
    )
    print("✓ RemoteDesktopEnv created successfully!")
    print(f"  - Env Port: {env.env_port}")
    print(f"  - Manager Port: {env.manager_port}")
    
    # 测试获取 VM 信息
    # 注意: vm_platform 和 vm_screen_size 可能需要在 reset() 之后才能访问
    # 因为 DesktopEnv 的 controller 属性可能在 reset() 时才初始化
    print(f"\n[测试 VM 信息]")
    print("  注意: 这些属性可能需要在 reset() 之后才能访问")
    
    try:
        vm_platform = env.vm_platform
        print(f"  ✓ VM Platform: {vm_platform}")
    except Exception as e:
        error_msg = str(e)
        if "controller" in error_msg.lower():
            print(f"  ⚠ VM Platform 获取失败: {error_msg}")
            print(f"     提示: 这通常是因为 DesktopEnv 的 controller 属性未初始化")
            print(f"           可能需要在调用 reset() 之后才能访问此属性")
        else:
            print(f"  ✗ VM Platform 获取失败: {e}")
    
    try:
        vm_screen_size = env.vm_screen_size
        print(f"  ✓ VM Screen Size: {vm_screen_size}")
    except Exception as e:
        error_msg = str(e)
        if "controller" in error_msg.lower():
            print(f"  ⚠ VM Screen Size 获取失败: {error_msg}")
            print(f"     提示: 这通常是因为 DesktopEnv 的 controller 属性未初始化")
            print(f"           可能需要在调用 reset() 之后才能访问此属性")
        else:
            print(f"  ✗ VM Screen Size 获取失败: {e}")
    
    # 测试 get_pid (返回所有 env 的 PID 列表)
    try:
        pids = env.get_pid
        print(f"  ✓ All Env PIDs: {pids}")
        if pids:
            print(f"     (当前有 {len(pids)} 个环境在运行)")
    except Exception as e:
        print(f"  ✗ VM PID 获取失败: {e}")
    
    # 清理当前环境
    print(f"\n[清理当前环境]")
    env.close()
    print("  ✓ Environment closed successfully")
    
    # 保存 manager_port 以便后续清理所有环境
    manager_port = env.manager_port
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    # 如果出错，使用默认的 manager_port
    manager_port = 9001

# 清理所有环境（通过 manager）
print("\n" + "=" * 60)
print("[清理所有环境]")
try:
    import requests
    base_url = "http://120.255.0.146"
    clean_url = f"{base_url}:{manager_port}/clean"
    response = requests.post(clean_url, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print("  ✓ 所有环境已清理")
        else:
            print(f"  ⚠ 清理环境时出现问题: {result.get('message', 'Unknown error')}")
    else:
        print(f"  ✗ 清理环境失败: HTTP {response.status_code}")
        print(f"     响应: {response.text[:200]}")
except Exception as e:
    print(f"  ✗ 清理所有环境失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
# 关闭 Ray
ray.shutdown()
