"""
This is the server side for the StreamMUSE end to end system.
"""

import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
from contextlib import asynccontextmanager

# 代码结构说明（简短）:
# - lifespan: FastAPI 启动时加载推理引擎 (InferenceEngineStanley)，并初始化注入状态。
# - /inject_music: 将指定 MIDI 文件（mel 和 acc）读入并注入到推理引擎的历史中（供后续推理使用）。
# - /injection_status: 查询当前注入状态。
# - /generate_accompaniment: 接收客户端 melody note events，调用 inference_engine.generate_accompaniment 返回生成的伴奏。
#
# 如果你希望 "acc"（伴奏）来自不同来源（例如数据库、另一个目录、网络服务或不同命名规则的文件），
# 请在下面的注入处理逻辑中查找标注为 "CHANGE ACC SOURCE HERE" 的注释位置进行修改。

# from app.inference_engines.transformer_engine import TransformerInferenceEngine
from app.inference_engines.transformer_engine_stanley import InferenceEngineStanley


class MelodyNoteEvent(BaseModel):
    pitch: int
    tick: int
    duration: int


class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float
    generation_length_frames: int = None


class AccompanimentNoteEvent(BaseModel):
    pitch: int
    tick: int
    duration: int
    program: int


class Timings(BaseModel):
    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float


class AccompanimentResponse(BaseModel):
    accompaniment: list[AccompanimentNoteEvent]
    timings: Timings
    generation_start_tick: int


# For injection requests
class InjectionRequest(BaseModel):
    injection_file_path: str
    injection_length_ticks: int


class InjectionResponse(BaseModel):
    success: bool
    message: str
    injection_length_ticks: int
    melody_notes_injected: int
    accompaniment_notes_injected: int


# app = FastAPI(title='StreamMUSE Inference Server')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler: 在应用启动时加载模型
    """
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Fatal Error: CHECKPOINT_PATH environment variable is not set")
        print("Please run the server like: CHECKPOINT_PATH=path/to/model.ckpt uvicorn ...")
        exit()

    # Get model parameters from environment variables with defaults
    try:
        model_max_seq_len_frames = int(os.getenv("MODEL_MAX_SEQ_LEN_FRAMES", 96))
        model_size = os.getenv("MODEL_SIZE", "0.12B")
    except ValueError:
        print("Fatal Error: Invalid integer value for model parameters in environment variables.")
        exit()

    # 验证 model_size 是否有效
    valid_model_sizes = ["small", "0.12B", "0.25B", "0.5B"]
    if model_size not in valid_model_sizes:
        print(f"Fatal Error: Invalid MODEL_SIZE '{model_size}'. Valid options: {valid_model_sizes}")
        exit()

    global inference_engine, injection_state
    try:
        print(f"Loading model from {checkpoint_path}...")
        print(f"Using Model Max Sequence Length (Frames): {model_max_seq_len_frames}")
        inference_engine = InferenceEngineStanley(
            checkpoint_path=checkpoint_path,
            model_size=model_size,
            model_max_seq_len_frames=model_max_seq_len_frames,
        )
        print("Inference engine loaded successfully.")

        # 初始化注入状态
        injection_state = {
            "is_injected": False,
            "injection_length_ticks": 0,
            "injection_file_path": None,
        }

    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        exit()
    yield
    # 这里可以添加关闭/清理逻辑（可选）


app = FastAPI(title="StreamMUSE Inference Server", lifespan=lifespan)


# 添加注入端点
@app.post("/inject_music", response_model=InjectionResponse)
async def inject_music(request: InjectionRequest):
    """
    注入一段音乐到推理引擎的历史中
    """
    global injection_state

    if not inference_engine:
        return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})

    try:
        # 检查文件是否存在
        melody_file_path = request.injection_file_path
        if not os.path.exists(melody_file_path):
            return InjectionResponse(
                success=False,
                message=f"Injection file not found: {melody_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # 自动推导伴奏文件路径
        # 比如将 input/mel/001.mid 转换为 input/acc/001.mid
        accompaniment_file_path = "acc-poly-pattern/c_major_poly_stride.mid"
        # acc_template_dir = "acc-poly-pattern/"
        # acc_template_list = sorted(os.listdir(acc_template_dir))
        # i = 0
        # accompaniment_file_path = os.path.join(acc_template_dir, acc_template_list[i])
        print(f"Using accompaniment file: {accompaniment_file_path}")
        # ----- CHANGE ACC SOURCE HERE -----
        # 位置说明：上面使用简单的字符串替换从 melody 文件路径推导 accompaniment 文件路径。
        # 如果你的伴奏来自其他来源，这里是修改的关键位置：
        # - 替换逻辑（例如不同目录结构或不同后缀）
        # - 从数据库/对象存储/网络下载伴奏并写到临时文件，然后将 accompaniment_file_path 指向该临时文件
        # - 或者直接读取伴奏数据并把它转换为与 midi_to_note 相同的内部表示（避免文件系统）
        # 举例（伪代码）:
        # if use_db:
        #     accompaniment_file_path = fetch_acc_from_db(melody_file_path)
        # elif use_remote:
        #     accompaniment_file_path = download_acc_for(melody_file_path)
        # 记得在修改后保持：
        # - 返回的 accompaniment_file_path 指向一个存在的 MIDI 文件（或更改下面的读取逻辑以直接接受数据流）
        # - 保持对不存在文件的错误处理
        # ----------------------------------

        # 验证路径替换是否成功
        if accompaniment_file_path == melody_file_path:
            return InjectionResponse(
                success=False,
                message=f"无法推导伴奏文件路径，旋律文件路径应包含 '/mel/' 目录: {melody_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # 检查伴奏文件是否存在
        if not os.path.exists(accompaniment_file_path):
            return InjectionResponse(
                success=False,
                message=f"Accompaniment file not found: {accompaniment_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # 清除现有历史
        inference_engine.clear_history()

        # 设置注入偏移
        inference_engine.set_injection_offset(request.injection_length_ticks)

        # 读取注入文件
        from app.midi_input_script import midi_to_note

        # 读取旋律和伴奏
        melody_notes, _, _ = midi_to_note(melody_file_path, max_tick=request.injection_length_ticks)

        accompaniment_notes, _, _ = midi_to_note(accompaniment_file_path, max_tick=request.injection_length_ticks)
        # ----- NOTE -----
        # 这里使用的是 `midi_to_note` 将 MIDI 文件解析为内部 note dict 列表。
        # 如果你选择不通过文件系统获得伴奏（例如从 DB/网络直接返回 note 列表），
        # 可在此处直接使用该列表并跳过 midi_to_note 调用。保持与下面的注入兼容：
        # inference_engine.accompaniment_history 应当接收与 melody_notes 相同格式的 note dict 列表。
        # ----------------

        # 过滤只保留指定长度内的音符
        melody_notes = [n for n in melody_notes if n["tick"] < request.injection_length_ticks]
        accompaniment_notes = [n for n in accompaniment_notes if n["tick"] < request.injection_length_ticks]

        # 注入到引擎历史中
        inference_engine.melody_history.extend(melody_notes)
        inference_engine.accompaniment_history.extend(accompaniment_notes)
        # ----- CHANGE ACC SOURCE HERE -----
        # 位置说明：如果影片来源不是文件并且你在上面已经构建了 accompaniment_notes 列表，
        # 请确保这里使用正确的变量（例如来自 DB 的 `acc_notes`），并将它扩展到
        # `inference_engine.accompaniment_history`。这是将伴奏数据传递给推理引擎的最终位置。
        # ----------------------------------

        # 更新注入状态
        injection_state = {
            "is_injected": True,
            "injection_length_ticks": request.injection_length_ticks,
            "injection_file_path": request.injection_file_path,
        }

        print(f"注入完成: {len(melody_notes)} 个旋律音符, {len(accompaniment_notes)} 个伴奏音符")

        return InjectionResponse(
            success=True,
            message="Music injected successfully",
            injection_length_ticks=request.injection_length_ticks,
            melody_notes_injected=len(melody_notes),
            accompaniment_notes_injected=len(accompaniment_notes),
        )

    except Exception as e:
        return InjectionResponse(
            success=False,
            message=f"Error injecting music: {str(e)}",
            injection_length_ticks=0,
            melody_notes_injected=0,
            accompaniment_notes_injected=0,
        )


# 添加获取注入状态的端点
@app.get("/injection_status")
async def get_injection_status():
    """
    获取当前注入状态
    """
    global injection_state
    return injection_state


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest):
    """
    Receive list of note events from client
    Returns generated list of accompaniment events with timing info.
    """
    request_arrival_time = time.perf_counter()

    if not inference_engine:
        return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})

    melody_notes_dicts = [note.dict() for note in request.melody_notes]

    (
        accompaniment_dicts,
        preprocess_start_time,
        inference_start_time,
        inference_end_time,
        postprocess_start_time,
    ) = inference_engine.generate_accompaniment(
        melody_notes_dicts,
        generation_start_tick=request.generation_start_tick,
        generation_length_frames=request.generation_length_frames,  # may be None
    )
    # ----- NOTE -----
    # 这里 `accompaniment_dicts` 是推理结果：一组伴奏音符的字典列表。
    # 如果你改变了伴奏的来源（注入历史或实时替换），推理引擎会在 generate_accompaniment 内
    # 利用 `inference_engine.accompaniment_history` 和 `inference_engine.melody_history` 来决定上下文。
    # 若需在推理时动态替换伴奏上下文，可以在 InferenceEngineStanley 的实现中添加钩子或参数，
    # 并在此处传入所需的上下文/标志。
    # ----------------
    print(f"Using generation length (frames): {request.generation_length_frames}")

    response_output_time = time.perf_counter()

    return AccompanimentResponse(
        accompaniment=accompaniment_dicts,
        timings=Timings(
            request_arrival_time=request_arrival_time,
            response_output_time=response_output_time,
            preprocess_start_time=preprocess_start_time,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            postprocess_start_time=postprocess_start_time,
        ),
        generation_start_tick=request.generation_start_tick,
    )


@app.post("/clear_history")
async def clear_history():
    """
    清除历史和注入状态
    """
    global injection_state

    if inference_engine:
        inference_engine.clear_history()
        inference_engine.set_injection_offset(0)  # 重置注入偏移
        injection_state = {
            "is_injected": False,
            "injection_length_ticks": 0,
            "injection_file_path": None,
        }
        return {"message": "History and injection state cleared successfully."}
    return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})
