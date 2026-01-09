Subject: README.md

# Vehicle Price Prediction Model

This project predicts vehicle sold prices using structured vehicle data and a LightGBM regression model.
The model learns the ratio of **Sold_Amount / NewPrice** and applies it to unseen vehicles.
It also provides **SHAP-based explainability** and **MAPE analysis by price bands**.
---

## 📦 Project Structure

```
vehicle-price-model/
│
├── data/
│   ├── DatiumTrain.rpt
│   ├── DatiumTest.rpt
│
├── src/
│   ├── train_model.py
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/vehicle-price-model.git
cd vehicle-price-model
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Model

```bash
python src/train_model.py
```

---

## 📊 Outputs

* Model predictions
* Performance metrics (MAE, RMSE, MAPE)
* SHAP feature importance plots

---

## 📌 Notes

* Large CSV files are excluded from Git
* Update file paths inside scripts if needed

---

## 📄 License

MIT License
