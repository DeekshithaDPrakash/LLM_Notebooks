from transformers import AutoModelForCausalLM, AutoTokenizer

#--------------Load Model and Check Configurations-------------------

# 모델과 토크나이저 로드
model_name = "gpt-3.5-turbo"  # 모델 이름
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델의 설정 정보 확인
print("Model Configuration:")
print(model.config)

#---------------Check for Built-in Prompt Generation---------------

# 프롬프트 템플릿을 유추할 수 있는 추가 정보
if hasattr(model.config, "prompt_template"):
    print("Prompt Template:")
    print(model.config.prompt_template)
else:
    print("No explicit prompt template found in the config.")

# 프롬프트 템플릿 직접 확인
if hasattr(model, "generate_prompt"):
    prompt = model.generate_prompt("input_text_example")
    print("Generated Prompt:")
    print(prompt)
else:
    print("The model does not have a built-in prompt generation function.")
  
#---------------------tokenize and generate output---------------------------------------
# 간단한 프롬프트 입력
input_text = "Translate the following sentence to French: 'Hello, how are you?'"

# 토크나이저로 입력 전처리
inputs = tokenizer(input_text, return_tensors="pt")

# 모델 출력 확인
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model Output:")
print(response)

#----------------download and extract model config from huggingface------------------------
from huggingface_hub import hf_hub_download

# 저장소에서 구성 파일 다운로드
repo_id = "your_model_repository"  # 모델 저장소 이름
config_file = hf_hub_download(repo_id, filename="config.json")

# 구성 파일 읽기
import json

with open(config_file, "r") as f:
    config_data = json.load(f)

# 프롬프트 템플릿 확인
if "prompt_template" in config_data:
    print("Prompt Template:")
    print(config_data["prompt_template"])
else:
    print("No prompt template found in the configuration.")

