"""
M2A (Melody-to-Accompaniment) Transformer 推理脚本

这个脚本用于：
1. 加载训练好的M2A模型
2. 对输入的MIDI文件进行旋律到伴奏的生成
3. 支持从头生成或基于提示生成
4. 输出生成的MIDI文件
"""

from m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
from preprocess.preprocess_midi2pt_dataset import preprocess_midi, DURATION_TEMPLATES
import torch
import pretty_midi
import os
import argparse
from typing import Literal


def decode_output(outputs, save_path, tempo=120.0, prompt=True, single=False):
    """
    将模型输出的token序列解码为MIDI文件

    Args:
        outputs: 模型输出的token序列列表，每个元素形状为 [1, subseq_len]
        save_path: 保存MIDI文件的路径
        tempo: MIDI文件的BPM速度，默认120
        prompt: 是否包含提示（未使用）
        single: 是否为单声部模式（影响时间步计算）

    处理流程：
        1. 创建空白MIDI对象
        2. 遍历每个时间步的输出token
        3. 将token对解码为 (program, pitch_duration)
        4. 从pitch_duration中提取音高和时长
        5. 创建MIDI音符并添加到对应乐器轨道
    """
    # 创建MIDI对象，设置初始速度
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # 计算每个时间步的长度（四分音符 = 1个时间步）
    time_step_length = 60.0 / tempo / 4

    # 确保outputs是元组格式
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    # 遍历每个输出序列（通常只有一个）
    for output in outputs:
        # 存储不同program的乐器实例
        instrument_map: dict[Literal[0, 1], pretty_midi.Instrument] = {}

        # 遍历每个时间步
        for time_step, data in enumerate(output):
            # 移除batch维度：[1, subseq_len] -> [subseq_len]
            content = data.squeeze(0)

            # 计算实际时间步（single模式不需要除2）
            time_step = time_step if single else time_step // 2
            start_time = time_step * time_step_length

            # 每2个token组成一个音符事件：(program, pitch_duration)
            for i in range(0, len(content), 2):
                program = int(content[i].item())

                # 遇到序列结束标记，停止处理
                if program == EOS_TOKEN:
                    break

                # 检查是否有完整的token对
                if i + 1 >= len(content):
                    print("Incomplete note @", time_step, i)
                    break

                # 获取pitch_duration token并解码
                pitch_duration = int(content[i + 1].item())
                pitch_duration = pitch_duration - 2  # 减去编码时的偏移

                # 从组合值中提取音高和时长
                # 编码公式：pitch + duration * 128
                pitch = pitch_duration % 128
                duration = pitch_duration // 128
                print(f"Note @ {time_step}, {i}: program={program}, pitch={pitch}, duration={duration}")
                # 验证program值（0=伴奏，1=旋律）
                if program != 0 and program != 1:
                    print("Invalid program:", program, "@", time_step, i)
                    break

                # 验证音高范围
                if pitch < 0 or pitch >= 128:
                    print("Invalid pitch:", pitch, "@", time_step, i)
                    break

                # 验证时长索引
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print("Invalid duration:", duration, "@", time_step, i)
                    break

                # 计算音符结束时间
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time

                # 创建乐器实例（如果不存在）
                if program not in instrument_map:
                    if program == 0:
                        # Program 0: 吉他（伴奏）
                        inst = pretty_midi.Instrument(program=24, name="Guitar")
                    else:  # program == 1
                        # Program 1: 钢琴（旋律）
                        inst = pretty_midi.Instrument(program=0, name="Piano")
                    instrument_map[program] = inst
                    midi.instruments.append(inst)

                # 添加音符到对应乐器
                inst = instrument_map[program]
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))

    # 创建保存目录并写入MIDI文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def decompress(model, byte_arr_mel, byte_arr_acc):
    """
    将预处理的字节数组转换为模型输入格式

    Args:
        model: M2A模型实例
        byte_arr_mel: 旋律数据的字节数组
        byte_arr_acc: 伴奏数据的字节数组

    Returns:
        tuple: (x_mel, x_acc) 预处理后的张量
               - x_mel: 旋律数据，形状 [1, seq_len, subseq_len]
               - x_acc: 伴奏数据，形状 [1, seq_len, subseq_len]
    """
    # 转换为张量并添加batch维度
    x = torch.tensor(byte_arr_mel).unsqueeze(0).cuda()
    y = torch.tensor(byte_arr_acc).unsqueeze(0).cuda()

    # 通过模型的预处理函数进行格式转换
    # pitch_shift=0表示不进行音高偏移
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), y=y)


def continuation(model, midi_path, prompt_length=100, generation_length=384, temperature=1.0, n_samples=1, gt_mel=True):
    """
    执行旋律到伴奏的生成任务

    Args:
        model: 训练好的M2A模型
        midi_path: 输入MIDI文件路径（旋律文件）
        prompt_length: 提示长度（帧数），0表示从头生成
        generation_length: 生成长度（帧数）
        temperature: 采样温度，0表示贪心采样，>0表示随机采样
        n_samples: 生成样本数量
        gt_mel: 是否使用ground truth旋律作为条件
    """
    # 验证文件存在性
    if os.path.isfile(midi_path):
        pass
    else:
        print(f"Error: {midi_path} is not a valid file.")
        return

    # 预处理MIDI文件为字节数组
    # 4表示子序列长度参数
    byte_arr_mel = preprocess_midi(midi_path, 4)
    byte_arr_acc = preprocess_midi(midi_path.replace("mel", "acc"), 4)

    # 检查预处理结果
    if byte_arr_mel is None:
        print(f"Error: preprocess_midi returned None for mel file: {midi_path}")
        return
    if byte_arr_acc is None:
        print(f"Error: preprocess_midi returned None for acc file: {midi_path.replace('mel', 'acc')}")
        return

    # 解压缩为模型输入格式
    x_mel, x_acc = decompress(model, byte_arr_mel[0], byte_arr_acc[0])

    # 确定生成起始位置
    if prompt_length == 0:
        # 从头生成：找到第一个有效帧作为起始点
        B, S, L = x_mel.shape
        valid = (x_mel != EOS_TOKEN) & (x_mel != PAD_TOKEN)  # [B, S, L]，标记有效token
        frame_has = valid.any(dim=2)  # [B, S]，标记有效帧
        idx = torch.arange(S, device=x_mel.device).unsqueeze(0).expand(B, S)  # [B, S]
        idx_masked = torch.where(frame_has, idx, torch.full_like(idx, S))  # [B, S]
        first_timestep = idx_masked.min(dim=1).values
    else:
        # 基于提示生成：从第1帧开始
        first_timestep = 1

    # 可选：保存原始旋律MIDI文件
    if not gt_mel:
        decode_output(
            [x_mel[:, i, :] for i in range(x_mel.shape[1])],
            f"temp/{model.save_name}/{os.path.basename(midi_path)}_originalmelody.mid",
            single=True,
            tempo=90.0,
        )

    # 准备数据切片
    x_mel_gt = x_mel.clone()  # 保存完整旋律作为ground truth
    x_mel_gt = x_mel_gt[:, first_timestep - 1 + prompt_length :]  # 提取未来旋律
    x_mel = x_mel[:, :prompt_length]  # 提取提示旋律
    x_acc = x_acc[:, :prompt_length]  # 提取提示伴奏

    # 获取数据维度信息
    batch_size, seq_len, subseq_len = x_mel.shape  # 例：[1, 150, 8]

    # 交错排列旋律和伴奏：[acc_0, mel_0, acc_1, mel_1, ...]
    stacked = torch.stack([x_acc, x_mel], dim=2)  # [B, S, 2, L]
    x = stacked.view(batch_size, seq_len * 2, subseq_len)  # [B, S*2, L]

    # 可选：保存提示部分的MIDI文件
    if prompt_length != 0:
        decode_output(
            [x[:, i, :] for i in range(x.shape[1])],
            f"temp/{model.save_name}/{os.path.basename(midi_path)}_promptlen{prompt_length}.mid",
            tempo=90.0,
        )

    # 执行生成过程
    with torch.no_grad():
        # 复制数据以生成多个样本
        x = x.repeat(n_samples, 1, 1)  # [n_samples, S*2, L]
        x_mel_gt = x_mel_gt.repeat(n_samples, 1, 1)  # [n_samples, remaining_S, L]

        import time

        start_time = time.time()

        # 根据提示长度选择生成策略
        if prompt_length == 0:
            # 从头生成：只使用旋律信息
            output = model.global_sampling_from_scratch(
                x_mel_gt, temperature=temperature, max_seq_len=generation_length
            )
        else:
            # 基于提示生成：使用提示和旋律信息
            output = model.global_sampling(
                x, x_mel_gt=x_mel_gt if gt_mel else None, temperature=temperature, max_seq_len=generation_length
            )

        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f} seconds")

    # 保存生成结果
    for i in range(n_samples):
        # 提取第i个样本的输出
        # output是列表，每个元素形状为 [n_samples, n*subseq_len]
        # 提取第i个样本：[n_samples, n*subseq_len] -> [1, n*subseq_len]
        output_i = [output[j][i : i + 1, :] for j in range(len(output))]

        # 解码并保存MIDI文件
        decode_output(
            output_i,
            f"temp/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_{i}.mid",
            tempo=90.0,
        )


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="M2A Transformer推理：将旋律转换为伴奏")

    parser.add_argument("--model_path", type=str, help="模型检查点文件路径")
    parser.add_argument("--prompt_len", type=int, default=150, help="提示长度（帧数）")
    parser.add_argument("--n_samples", type=int, default=1, help="生成样本数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")

    args = parser.parse_args()

    # 设置模型路径（硬编码，可通过命令行参数覆盖）

    # 添加模块搜索路径
    import sys

    sys.path.insert(0, "/home/ubuntu/ugrip/yuanhsin/Training-Framework")

    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # 导入并加载模型
    # from src.model.old_pt_m2a_new_.model import OldPtM2ANew
    # model_path = args.model_path
    # model_path = "/home/ubuntu/ugrip/yuanhsin/Training-Framework/output/m2a_new_Yuan/target_length=384/0.0.6/checkpoints/val/epoch=9-val/epoch/loss=0.3673.ckpt"
    # model = OldPtM2ANew.load_from_checkpoint(checkpoint_path=model_path, map_location=device)
    # 导入并加载模型
    from src.model.old_pt_m2a_transformer_with_att.model import OldPtM2ATransformerWithAttention

    model_path = args.model_path
    model_path = "/home/ubuntu/ugrip/yuanhsin/Training-Framework/output/m2a_with_att/dropout_prob=0.3-target_length=384/0.0.0/checkpoints/epoch=743-step=150288.ckpt"
    model = OldPtM2ATransformerWithAttention.load_from_checkpoint(checkpoint_path=model_path, map_location=device)
    # 设置模型属性和状态
    model.save_name = os.path.basename(model_path)  # 用于生成文件名
    model.cuda()  # 移动到GPU
    model.eval()  # 设置为评估模式

    # 批处理输入目录中的所有MIDI文件
    for midi in os.listdir("./input/mel"):
        if midi.endswith("mid"):
            midi = os.path.join("./input/mel", midi)
            # 对每个MIDI文件执行旋律到伴奏的生成
            continuation(
                model,
                midi,
                temperature=args.temperature,
                generation_length=100,  # 生成384帧
                n_samples=args.n_samples,
                prompt_length=args.prompt_len,
                gt_mel=True,  # 使用ground truth旋律作为条件
            )
