"""
Remote GUI Agent Environment Worker for distributed OSWorld training.

This module provides EnvWorkerRemote class that communicates with a remote OSWorld
server via HTTP API, enabling distributed training across multiple machines.

Usage:
    from verl.trainer.gui_agent_remote import EnvWorkerRemote
    
    # Create a remote worker
    worker = EnvWorkerRemote.remote(
        worker_idx=0,
        max_steps=15,
        config=config,
        remote_env_config={
            "base_url": "http://192.168.1.100",
            "manager_port": 8000,  # or use env_port directly
            "vm_name": "Ubuntu.qcow2",
            "os_type": "Ubuntu",
        }
    )
    
    # Reset with task config
    result = ray.get(worker.reset.remote(task_config))
    
    # Step through the environment
    result = ray.get(worker.step.remote(prediction))
    
    # Evaluate
    reward = ray.get(worker.evaluate.remote())
"""

import ast
import re
import math
import time
import requests
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import ray
import os
from io import BytesIO
import base64
import datetime
import traceback
import torch
from qwen_vl_utils import process_vision_info


# ============================================================================
# Constants and Prompts
# ============================================================================

uitars_system_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


# ============================================================================
# HTTP API Utilities
# ============================================================================

def request_api_wrapper(
    url: str,
    data: Optional[Dict] = None,
    try_max_times: int = 5,
    method: str = "POST",
    timeout: int = 360
) -> Dict[str, Any]:
    """Synchronous request API wrapper for remote OSWorld server.
    
    Args:
        url: The API endpoint URL.
        data: JSON data to send with the request.
        try_max_times: Maximum number of retry attempts.
        method: HTTP method (POST or GET).
        timeout: Request timeout in seconds.
    
    Returns:
        Response JSON if successful.
    
    Raises:
        Exception: If all retry attempts fail.
    """
    headers = {
        "Content-Type": "application/json",
    }
    for attempt in range(try_max_times):
        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            response_json = response.json()
            # Check success field if it exists, otherwise assume success
            if "success" in response_json:
                is_success = response_json.get("success")
                if not is_success:
                    message = response_json.get("message", None)
                    print(f"API execution error: {message}")
                else:
                    return response_json
            else:
                # No success field, assume the request was successful
                return response_json
        except requests.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{try_max_times}): {e}")
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}/{try_max_times}): {e}")
        time.sleep(1)

    raise Exception(
        f"Request error for {try_max_times} times. Please check the API server at {url}"
    )


# ============================================================================
# Remote Desktop Environment
# ============================================================================

class RemoteDesktopEnv:
    """Remote Desktop Environment that communicates with a remote OSWorld server via HTTP API.
    
    This class provides the same interface as DesktopEnv but delegates all operations
    to a remote server, enabling distributed training across multiple machines.
    
    Args:
        base_url: Base URL of the remote OSWorld server (e.g., "http://192.168.1.100").
        env_port: Port of the environment API. If None, use manager_port to create one.
        manager_port: Port of the environment manager API (used when env_port is None).
        vm_name: Name of the virtual machine image file.
        action_space: Action space type ("pyautogui" or "computer_13").
        screen_size: Screen resolution as (width, height).
        headless: Whether to run VM in headless mode.
        require_a11y_tree: Whether to include accessibility tree in observations.
        os_type: Operating system type ("Ubuntu" or "Windows").
        connect_max_try: Maximum connection retry attempts.
    """
    
    def __init__(
        self,
        base_url: str,
        env_port: Optional[int] = None,
        manager_port: Optional[int] = None,
        vm_name: str = "Ubuntu.qcow2",
        action_space: str = "pyautogui",
        screen_size: Tuple[int, int] = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = False,
        os_type: str = "Ubuntu",
        connect_max_try: int = 5,
    ):
        self.base_url = base_url
        self.env_port = env_port
        self.manager_port = manager_port
        self.connect_max_try = connect_max_try
        self.action_history = []  # Track actions for evaluation

        # Create env api by manager if env_port not provided
        if env_port is None:
            assert manager_port is not None, "Must provide either env_port or manager_port"
            self.use_api_manager = True
            create_url = f"{base_url}:{manager_port}/create_env_api"
            response = request_api_wrapper(create_url, try_max_times=self.connect_max_try)
            self.env_id = response["env_id"]
            self.env_port = response["port"]
            time.sleep(7)
            print(f"Created env API with env_id={self.env_id}, port={self.env_port}")
        else:
            self.use_api_manager = False
            self.env_id = None

        # Create environment on remote server
        data = {
            "vm_name": vm_name,
            "action_space": action_space,
            "screen_width": screen_size[0],
            "screen_height": screen_size[1],
            "headless": headless,
            "require_a11y_tree": require_a11y_tree,
            "os_type": os_type
        }
        start_url = f"{base_url}:{self.env_port}/start"
        response = request_api_wrapper(start_url, data, try_max_times=self.connect_max_try)
        print("Remote environment created successfully")

    def reset(self, task_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reset the environment with a new task configuration.
        
        Args:
            task_config: Task configuration dictionary containing task details.
        
        Returns:
            Observation dictionary with screenshot and other info, or None on failure.
        """
        print("Resetting remote environment...")
        self.action_history.clear()
        reset_url = f"{self.base_url}:{self.env_port}/reset"
        try:
            response = request_api_wrapper(
                reset_url, {"task_config": task_config}, try_max_times=10
            )
            obs = response["obs"]
            obs["screenshot"] = base64.b64decode(obs["screenshot"])  # base64 -> bytes
            print("Remote environment reset done.")
            return obs
        except Exception as e:
            print(f"Remote reset failed: {e}")
            return None

    def step(
        self,
        action: str,
        pause: float = 2
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action in the environment.
        
        Args:
            action: The action to execute (pyautogui code string or special action).
            pause: Pause duration after action execution.
        
        Returns:
            Tuple of (observation, reward, done, info).
        """
        self.action_history.append(action)
        data = {
            "action": action,
            "pause": pause
        }
        step_url = f"{self.base_url}:{self.env_port}/step"
        try:
            response = request_api_wrapper(step_url, data, try_max_times=self.connect_max_try)
            obs = response["obs"]
            obs["screenshot"] = base64.b64decode(obs["screenshot"])  # base64 -> bytes
            return obs, response["reward"], response["done"], response["info"]
        except Exception as e:
            print(f"Remote step failed: {e}")
            return {"screenshot": None}, -1, True, {"error": str(e)}

    def evaluate(self) -> float:
        """Evaluate the current task completion.
        
        Returns:
            Evaluation metric (float between 0 and 1).
        """
        evaluate_url = f"{self.base_url}:{self.env_port}/evaluate"
        try:
            response = request_api_wrapper(
                evaluate_url, method='GET', try_max_times=self.connect_max_try
            )
            return response.get("metric", 0.0)
        except Exception as e:
            print(f"Remote evaluation failed: {e}")
            return 0.0

    def pause(self) -> None:
        """Pause the remote environment (no-op for remote env, server manages VM state)."""
        # Remote environment doesn't need pause/unpause as the server manages VM state
        pass

    def unpause(self) -> None:
        """Unpause the remote environment (no-op for remote env, server manages VM state)."""
        # Remote environment doesn't need pause/unpause as the server manages VM state
        pass

    def close(self) -> None:
        """Close the environment and release resources."""
        # Stop VM
        close_url = f"{self.base_url}:{self.env_port}/close"
        try:
            _ = request_api_wrapper(close_url, try_max_times=self.connect_max_try)
        except Exception as e:
            print(f"Failed to close remote environment: {e}")
        
        # Terminate env API if using manager
        if self.use_api_manager and self.env_id is not None:
            terminate_url = f"{self.base_url}:{self.manager_port}/terminate_env_api"
            data = {"env_id": self.env_id}
            try:
                _ = request_api_wrapper(terminate_url, data, try_max_times=self.connect_max_try)
            except Exception as e:
                print(f"Failed to terminate env API: {e}")

    def clean(self) -> Dict[str, Any]:
        """Clean all environments managed by the manager.
        
        This method calls the manager's /clean endpoint to terminate all
        environment APIs and their associated processes.
        
        Returns:
            Response dictionary with success status and message.
        """
        if self.manager_port is None:
            raise ValueError("manager_port is required to call clean()")
        
        clean_url = f"{self.base_url}:{self.manager_port}/clean"
        try:
            response = request_api_wrapper(clean_url, method='POST', try_max_times=self.connect_max_try)
            return response
        except Exception as e:
            print(f"Failed to clean all environments: {e}")
            return {"success": False, "message": str(e)}

    @property
    def get_pid(self) -> list:
        """Get the list of all VM process IDs from the manager.
        
        Note: The manager returns a list of all env process PIDs.
        If you need the PID for this specific env, you may need to match by env_id.
        
        Returns:
            List of process IDs (integers).
        """
        get_pid_url = f"{self.base_url}:{self.manager_port}/get_pid"
        response = request_api_wrapper(
            get_pid_url, method='GET', try_max_times=self.connect_max_try
        )
        # Server returns {"pid": [list of pids]}
        return response.get("pid", [])
    
    @property
    def vm_platform(self) -> str:
        """Get the VM platform type."""
        vm_platform_url = f"{self.base_url}:{self.env_port}/vm_platform"
        response = request_api_wrapper(
            vm_platform_url, method='GET', try_max_times=self.connect_max_try
        )
        return response["vm_platform"]

    @property
    def vm_screen_size(self) -> Dict[str, int]:
        """Get the VM screen size."""
        vm_screen_size_url = f"{self.base_url}:{self.env_port}/vm_screen_size"
        response = request_api_wrapper(
            vm_screen_size_url, method='GET', try_max_times=self.connect_max_try
        )
        return response["vm_screen_size"]


# ============================================================================
# Action Parsing Utilities (same as gui_agent.py)
# ============================================================================

# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def linear_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)

    return height, width 

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, model_type, max_pixels=16384*28*28, min_pixels=100*28*28):
    text = text.strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(origin_resized_height, origin_resized_width, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        elif "finished(content" in action_str:
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content
            
            # 使用正则表达式进行替换
            pattern = r"finished\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "finished(content='" + action_str + "')"
            
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}") 
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
        
        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"
            
            elif hotkey == "space":
                hotkey = " "
                
            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


# ============================================================================
# Remote Environment Worker
# ============================================================================

from ..utils.tokenizer import get_processor, get_tokenizer

import verl.utils.torch_functional as VF
from ..models.transformers.qwen2_vl import get_rope_index

@ray.remote(num_cpus=1)
class EnvWorkerRemote():
    """Remote Environment worker for GUI agent training.
    
    This worker manages a remote desktop environment via HTTP API
    and handles the interaction loop between the agent and the environment.
    
    Args:
        worker_idx: Unique index for this worker.
        max_steps: Maximum number of steps per episode.
        config: Configuration object containing model and training settings.
        remote_env_config: Configuration for remote environment.
            Required keys:
            - base_url: Base URL of the remote server (e.g., "http://192.168.1.100")
            Optional keys:
            - env_port: Port of the environment API (use this OR manager_port)
            - manager_port: Port of the environment manager
            - vm_name: VM image name, default "Ubuntu.qcow2"
            - os_type: OS type, default "Ubuntu"
            - connect_max_try: Max connection retry attempts, default 5
    """
    system_prompt = uitars_system_prompt
    
    ground_prompt = r"""Output only the coordinate of one point in your response. What element matches the following task: """

    def __init__(self, worker_idx, max_steps=15, config=None, remote_env_config=None):
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.config = config

        # self.tokenizer = get_tokenizer(
        #     config.worker.actor.model.model_path,
        #     trust_remote_code=config.worker.actor.model.trust_remote_code,
        #     use_fast=True,
        # )
        # self.processor = get_processor(
        #     config.worker.actor.model.model_path,
        #     trust_remote_code=config.worker.actor.model.trust_remote_code,
        #     use_fast=True,
        # )

        self.model = 'uitars'
        
        # Create remote environment
        assert remote_env_config is not None, "remote_env_config is required for EnvWorkerRemote"
        print(f'Worker {worker_idx}: Creating remote desktop_env at {remote_env_config.get("base_url")}')
        
        self.env = RemoteDesktopEnv(
            base_url=remote_env_config.get("base_url"),
            env_port=remote_env_config.get("env_port", None),
            manager_port=remote_env_config.get("manager_port", None),
            vm_name=remote_env_config.get("vm_name", "Ubuntu.qcow2"),
            action_space="pyautogui",
            screen_size=(1920, 1080),
            headless=True,
            os_type=remote_env_config.get("os_type", "Ubuntu"),
            require_a11y_tree=False,
            connect_max_try=remote_env_config.get("connect_max_try", 5),
        )

        self.is_init = False
        self.is_done = False
        self.max_steps = max_steps
        # self.parser = Qwen2VLParser()

        self.action_parse_res_factor = 1000
        self.model_type = "qwen25vl"
        self.max_pixels = 16384*28*28
        self.min_pixels = 100*28*28

        self.instruction = None
        self.task_config = None
        self.step_counter = 0
        self.history_images = []
        self.history_messages = []
    
        self.reset_train_tensors()

    def reset_train_tensors(self):
        # for training
        self.input_ids = torch.zeros((0,), dtype=torch.int64)
        self.labels = torch.full((0,), -100, dtype=torch.int64)
        self.attention_mask = torch.zeros((0, ), dtype=torch.int64)

        self.pixel_values = None
        self.image_grid_thw = None

    def load_content(self, content):
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            return ''.join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        
        raise ValueError(f"Unknown content type: {content}")
    
    def process_message(self, message):
        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True)

        input_ids = []
        labels = []
        attention_mask = []

        image_count = 0
        pixel_values = []
        image_grid_thw = []
        for turn_idx, msg in enumerate(message):
            role = msg['role']
            content = self.load_content(msg['content'])
            prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'

            cur_image_num = prompt.count("<|image_pad|>")                
            if cur_image_num > 0:
                result = processor(image_inputs[image_count:image_count+cur_image_num], [prompt], add_special_tokens=False, return_tensors="pt")
                image_count += cur_image_num
            else:
                result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
            
            cur_input_ids = result.pop('input_ids')[0]
            cur_attention_mask = result.pop('attention_mask')[0]
            if 'pixel_values' in result: # 10764, 1176
                pixel_values.append(result["pixel_values"])
            if 'image_grid_thw' in result:
                image_grid_thw.append(result["image_grid_thw"])
            

            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            if role in ["system", "user"]:
                labels.append(torch.full_like(cur_input_ids, -100))
            else:
                labels.append(cur_input_ids)

        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)  

        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)

        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

        if self.pixel_values is None:
            self.pixel_values = pixel_values
        else:
            if pixel_values is not None:
                self.pixel_values = torch.cat([self.pixel_values, pixel_values], dim=0)
        
        if self.image_grid_thw is None:
            self.image_grid_thw = image_grid_thw
        else:
            if image_grid_thw is not None:
                self.image_grid_thw = torch.cat([self.image_grid_thw, image_grid_thw], dim=0)
            

    def get_train_dict(self):
        position_ids = get_rope_index(
                self.processor,
                input_ids=self.input_ids,
                image_grid_thw=self.image_grid_thw,
                attention_mask=self.attention_mask,
            )
        
        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
                labels=self.labels
            )
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }
        if self.pixel_values is not None:
            multi_modal_inputs = dict()
            multi_modal_inputs['pixel_values'] = self.pixel_values
            multi_modal_inputs['image_grid_thw'] = self.image_grid_thw
            data['multi_modal_inputs'] = multi_modal_inputs
        return data
    
    def reset(self, task_config):

        self.instruction = task_config.get("instruction", None)
        self.task_config = task_config
        self.step_counter = 0
        self.is_done = False

        self.reset_train_tensors()

        trial_time = 0
        obs = None
        while trial_time < 8:
            try:
                obs = self.env.reset(task_config)
                if obs is not None:
                    break
            except Exception as e:
                print(f"Env reset exception: {e}")
                print('Env reset error: ', traceback.format_exc())
            trial_time += 1
        
        if trial_time >= 8 or obs is None:
            self.is_init = True
            self.is_done = True
            print('Env reset failed after 8 trials: ', task_config)
            return {
                "env_idx": self.worker_idx,
                "obs_messages": None,
                "is_done": self.is_done,
                'format_reward': 0.0
            }

        # self.agent.reset()
        self.is_init = True


        init_image = obs["screenshot"]
        # init_image = Image.open(BytesIO(init_image))

        image_base64 = base64.b64encode(BytesIO(init_image).getvalue()).decode("utf-8")

        init_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt.format(instruction=self.instruction)
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            }
        ]

        self.history_images = [init_image]
        self.history_messages = init_messages

        self.process_message(init_messages)

        # Remote env doesn't need pause, but call it for interface consistency
        self.env.pause()
        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }
    
    def step(self, prediction):

        self.is_init = False

        # 1. parse to structure output, might exception
        # 2. parse to pyautogui code, might excpetion
        # 3. env.step, might timeout
        # 4. check the output format(if format is wrong/timeout/exception, return DONE)

        origin_resized_height = obs_image_height = 1080
        origin_resized_width = obs_image_width = 1920

        try:
            parsed_responses = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                origin_resized_height,
                origin_resized_width,
                self.model_type,
                self.max_pixels,
                self.min_pixels
            )

            actions = []
            for parsed_response in parsed_responses:
                if "action_type" in parsed_response:
                    if parsed_response["action_type"] == FINISH_WORD:
                        actions = ['DONE']
                        break
                    
                    elif parsed_response["action_type"] == WAIT_WORD:
                        actions = ['WAIT']
                        break
                    
                    elif parsed_response["action_type"] == ENV_FAIL_WORD:
                        actions = ['FAIL']
                        break

                    elif parsed_response["action_type"] == CALL_USER:
                        actions = ['FAIL']
                        break

                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    obs_image_height,
                    obs_image_width,
                    False # input_swap = False, don't use pyperclip
                )
                actions.append(pyautogui_code)
            
            format_reward = 0.0
        except:
            print('Parse action error: ', prediction)
            print('Error traceback: ', traceback.format_exc())
            format_reward = -1.0
            actions = ['DONE'] # error output format, stop the trajectory immediately

        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

        # Remote env doesn't need unpause, but call it for interface consistency
        self.env.unpause()
        
        obs = None
        for action in actions:
            obs, reward, step_done, info = self.env.step(action, pause=0.5)
        
            if step_done:
                self.is_done = True
            
            self.step_counter += 1
            if self.step_counter == self.max_steps:
                self.is_done = True

            step_data = {
                "step_num": self.step_counter,
                "action": action,
                "reward": reward,
                "done": step_done,
                "info": info,
                "action_timestamp": action_timestamp,
            }
        
        # Remote env doesn't need pause, but call it for interface consistency
        self.env.pause()

        # Handle case where obs might be None (remote call failed)
        if obs is None:
            obs = {"screenshot": None}

        self.history_images.append(obs.get('screenshot'))
        self.history_messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": add_box_token(prediction)
            }]
        })

        if not self.is_done:
            if obs.get('screenshot') is None:
                self.is_done = True
                # failed to get screenshot
                self.process_message(self.history_messages[-1:]) # gpt answer only
                return {
                    'env_idx': self.worker_idx,
                    'obs_messages': None,
                    'is_done': self.is_done,
                    'format_reward': format_reward
                }

            image_base64 = base64.b64encode(BytesIO(obs['screenshot']).getvalue()).decode('utf-8')

            self.history_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            })
            self.process_message(self.history_messages[-2:]) # gpt answer + next image
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward
            }
        else:
            self.process_message(self.history_messages[-1:]) # gpt answer only
            return {
                'env_idx': self.worker_idx,
                'obs_messages': None,
                'is_done': self.is_done,
                'format_reward': format_reward
            }
            
    
    def evaluate(self):
        try:
            # Remote env doesn't need unpause, but call it for interface consistency
            self.env.unpause()
            # we dont care the env after evaluation, since the reset will destroy it and create new env.
            return self.env.evaluate()
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0
            
    
    def get_history_messages(self):
        return self.history_messages
    
    def get_history_images(self):
        return self.history_images
    
    def is_done(self):
        return self.is_done

    def is_init(self):
        return self.is_init
    
    def close(self):
        """Close the remote environment and release resources."""
        try:
            self.env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")

    def clean(self):
        """Clean the remote environment and release resources."""
        try:
            self.env.clean()
        except Exception as e:
            print(f"Error cleaning environment: {e}")