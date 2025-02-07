# **Fine-Tuning LLaMA for Question Answering (SQuAD Dataset)**  

## **Overview**  
The code is a fine-tuning pipeline for **LLMs on the SQuAD dataset**, optimized for **question-answering tasks** using:  
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning  
- **8-bit quantization** to reduce GPU memory usage  
- **Custom GPU memory tracking & early stopping mechanisms**  
- **Evaluation with F1 and Exact Match scores**  

This project explores **the challenges of fine-tuning LLaMA-1B** on resource-constrained hardware and demonstrates **scalable Q&A training techniques.**  

---

## **🚀 Key Features**  
✅ **Fine-tunes LLaMA-1B** on SQuAD v2 for QA tasks  
✅ **LoRA optimization** for lightweight training  
✅ **8-bit quantization** for efficient GPU memory usage  
✅ **Custom real-time monitoring** of training speed & memory  
✅ **Evaluation with F1 and EM (Exact Match) scoring**  
✅ **AWS GPU scaling (T4 x4) with gradient accumulation**  

---

## **📁 Project Structure**  

```bash
├── llm_qa_finetune/
│   ├── finetune.py               # Fine-tuning script (LoRA + Quantization)
│   ├── llama_squad_finetune.py   # End-to-end training pipeline
│   ├── model_setup.py            # Loads & configures LLaMA model
│   ├── process_data.py           # Prepares SQuAD dataset
│   ├── eval_2.py                 # Evaluates model using F1 & EM scores
│   ├── run_inference.py          # Runs inference on new questions
│   ├── test_model.py             # Tests the fine-tuned model with real data
│   ├── test_model_loading.py     # Debugging script to check model loading
│   ├── README.md                 # Project documentation (this file)
```

---

## **🛠️ Setup & Training**  

### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2️⃣ Prepare SQuAD Dataset**  
Ensure **SQuAD v2** dataset is downloaded and preprocessed.  
```bash
python process_data.py
```

### **3️⃣ Run Fine-Tuning**  
```bash
python llama_squad_finetune.py
```

💡 **To train on AWS GPU (T4 x4):**  
- Update `config.py` with the correct **batch size & learning rate** for larger models.  
- Ensure AWS permissions allow multiple GPUs.  

---

## **📊 Evaluation & Results**  

| Model  | Dataset  | F1 Score | EM Score |  
|--------|---------|---------|----------|  
| LLaMA-1B | SQuAD v2 | **82%** | **-** |  

✔️ **Final Performance:** Achieved **82% F1** on SQuAD using **LoRA fine-tuning**  
✔️ **Challenges:** Could not scale to **LLaMA-7B due to GPU limitations**  
✔️ **Solution:** Used **gradient accumulation, quantization, & early stopping**  
✔️ **Custom Monitoring:** Built real-time tracking for memory & training speed before switching to **Weights & Biases**  

---

## **🔎 Inference (Ask Questions)**  

```bash
python run_inference.py
```
Example:
```python
input_prompt = "What is the capital of France?"
result = run_inference(model_path="./llm_qa_finetune/fine_tuned_llama_squad", tokenizer_path="./llm_qa_finetune/fine_tuned_llama_squad", input_prompt)
print(result)
```

---

## **🔮 Next Steps**  
🔹 **Try LLaMA-7B & 12B** with optimized **dataset size & batch tuning**  
🔹 **Improve context retention** using better **prompt structuring**  
🔹 **Deploy model as an API for real-time Q&A applications**  

---

## **🔍 Additional Resources**  

- [LLaMA-1B Model Card](https://huggingface.co/meta-llama/Llama-3.1-1B-Instruct)  
- [SQuAD v2.0 Dataset](https://huggingface.co/datasets/SQuAD)  

---