# API 文档：Prompt Expansion

该接口允许你扩展一个文本提示（prompt），以生成更长或更详细的文本。

## 下载模型

download model from https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin to models

## 请求参数

| 参数名 | 类型   | 必填 | 描述                                |
| ------ | ------ | ---- | ----------------------------------- |
| prompt | string | 是   | 要扩展的文本提示（最大长度为 4096） |
| seed   | string | 是   | 扩展的种子（最大长度为 4096）       |

成功的响应将返回扩展后的文本。

| 字段名 | 类型   | 描述                          |
| ------ | ------ | ----------------------------- |
| result | string | 扩展后的文本（最大长度 4096） |

# API

http://127.0.0.1:8188/prompt_expansion
