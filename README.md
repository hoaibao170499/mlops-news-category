# **MLOps News Category Classification**

Hệ thống phân loại tin tức vào 4 chuyên mục: **atheism**, **religion.misc**, **graphics**, **space**.
Pipeline gồm tiền xử lý – huấn luyện – đánh giá – phục vụ mô hình bằng FastAPI – đóng gói Docker – theo dõi bằng MLflow.

---

## **1. Giới thiệu**

* Mục tiêu: phân loại văn bản tin tức vào đúng chuyên mục.
* Mô hình sử dụng: **Multinomial Naive Bayes**.
* Áp dụng đầy đủ quy trình **MLOps**: preprocessing, training, evaluation, API serving, tracking và containerization.

---

## **2. Bộ dữ liệu**

* Nguồn: **20 Newsgroups**, sử dụng 4 nhãn.
* Tổng số mẫu: **3387**
* Train/Test: **80/20**
* Tiền xử lý:

  * Lowercase
  * Loại bỏ số, ký tự đặc biệt, HTML tags
  * Tokenization
  * Stopwords removal
  * Lemmatization
  * Biểu diễn bằng **TF-IDF**

---

## **3. Kết quả mô hình**

| Metric            | Train | Test |
| ----------------- | ----: | ---: |
| Accuracy          |  0.95 | 0.91 |
| Precision (macro) |  0.94 | 0.89 |
| Recall (macro)    |  0.95 | 0.90 |
| F1-score (macro)  |  0.94 | 0.90 |

### **Confusion Matrix**

<img src="https://drive.google.com/file/d/1BEpAr7SwRW-wyXtq5uIK8K3wPHYVijbS/view?usp=drive_link" width="420"/>

---

## **4. FastAPI Serving**

Chạy API:

```bash
python scripts/api.py
```

### **Endpoint**

**POST** `/news/predict`
Request:

```json
{ "posts": "My point is that you set up your views as the only way to believe" }
```

Response:

```json
{ "predicted_news": "atheism" }
```

---

## **5. Docker & MLflow**

Build và chạy services:

```bash
docker compose up -d --build
```

* MLflow UI: **[http://localhost:5050](http://localhost:5050)**
* FastAPI: **[http://localhost:3000](http://localhost:3000)**

Huấn luyện mô hình:

```bash
python scripts/train.py
```

---
