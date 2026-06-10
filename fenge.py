import pandas as pd
import re


def process_reasoning_data(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确保必要的列存在
    required_columns = ['text_reasoning', 'image_reasoning', 'cross_modal_reasoning']
    for col in required_columns:
        if col not in df.columns:
            # 如果列不存在则创建
            df[col] = ""

    # 定义匹配模式，寻找以1)、2)、3)开头的条目
    pattern = r'(\d\)\s.*?)(?=\s*\d\)|$)'

    # 处理每一行
    for index, row in df.iterrows():
        # 获取text_reasoning列的内容
        reasoning_text = str(row['text_reasoning']).strip()

        # 找到所有匹配的部分
        matches = re.findall(pattern, reasoning_text, re.DOTALL)

        # 初始化三个列的内容
        text_content = ""
        image_content = ""
        cross_content = ""

        # 分配内容到对应变量
        for match in matches:
            if match.startswith('1)'):
                # 去除"1) "前缀
                text_content = match[2:].strip()
            elif match.startswith('2)'):
                # 去除"2) "前缀
                image_content = match[2:].strip()
            elif match.startswith('3)'):
                # 去除"3) "前缀
                cross_content = match[2:].strip()

        # 将处理后的内容赋值给对应列
        df.at[index, 'text_reasoning'] = text_content
        df.at[index, 'image_reasoning'] = image_content
        df.at[index, 'cross_modal_reasoning'] = cross_content

    # 保存处理后的CSV
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    input_csv = "./weibo_with_reasoning.csv"  # 输入文件路径
    output_csv = "./processed_weibo_reasoning.csv"  # 输出文件路径
    process_reasoning_data(input_csv, output_csv)
