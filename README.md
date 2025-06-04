# 📱 Ethiopian Bank App Review Scraper & Analyzer

This project is part of Omega Consultancy’s Week 2 Data Challenge and simulates a real-world consulting task for three major Ethiopian banks:

- **Commercial Bank of Ethiopia (CBE)**
- **Bank of Abyssinia (BOA)**
- **Dashen Bank**

We aim to scrape, clean, and analyze Google Play Store reviews to extract actionable insights about customer satisfaction, feature requests, and app performance.

---

## 🎯 Business Objective

Omega Consultancy advises fintech clients on improving customer experience and digital retention. As a data analyst, your objectives are:

- Scrape 400+ user reviews per bank from the Google Play Store.
- Preprocess reviews into clean, analysis-ready format.
- Analyze review **sentiment**, **themes**, and **complaints**.
- Store data in an Oracle-compatible structure.
- Deliver insights and data-driven recommendations.

---

## 📦 Project Structure

```
project-root/
│
├── scrape_google_play_reviews.py    # Scraping and preprocessing script
├── requirements.txt                 # Dependencies
├── README.md                        # Project overview and instructions
└── data/
    └── cleaned_bank_reviews_YYYY-MM-DD.csv   # Final output
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bank-review-scraper.git
cd bank-review-scraper
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### To scrape and clean data:

```bash
python scrape_google_play_reviews.py
```

- Scrapes 400+ reviews from each bank's official mobile app on the Google Play Store.
- Outputs a single cleaned CSV file: `cleaned_bank_reviews_YYYY-MM-DD.csv`

---

## 🔍 Methodology

### 🔹 Review Scraping

- **Library**: [`google-play-scraper`](https://pypi.org/project/google-play-scraper/)
- **Apps Targeted**:
  - Commercial Bank of Ethiopia: `com.ethiomobile.cbe`
  - Bank of Abyssinia: `com.bankofabyssinia.boaapp`
  - Dashen Bank: `com.mobin.dashenbank` (confirm package name)
- **Scrape Count**: 400+ reviews per app
- **Fields Extracted**:
  - `review`: Raw user review
  - `rating`: 1–5 star rating
  - `date`: Normalized to `YYYY-MM-DD`
  - `bank`: App/bank name
  - `source`: Hardcoded as `"Google Play"`

### 🔹 Preprocessing

- Remove duplicates (by review and bank).
- Drop reviews with missing fields.
- Export clean data as CSV.

---

## 📊 Example Output (CSV Preview)

| review                              | rating | date       | bank                        | source      |
| ----------------------------------- | ------ | ---------- | --------------------------- | ----------- |
| Love the app, but login fails often | 3      | 2024-06-03 | Commercial Bank of Ethiopia | Google Play |
| Fast transfers, but needs dark mode | 4      | 2024-06-02 | Dashen Bank                 | Google Play |
| Unstable and keeps crashing         | 2      | 2024-06-01 | Bank of Abyssinia           | Google Play |

---

## ✅ Deliverables (Task 1)

- [x] ✅ GitHub repo with code and `.gitignore`, `requirements.txt`, and `README.md`
- [x] ✅ Python script for scraping and preprocessing
- [x] ✅ At least **1,200 cleaned reviews** with <5% missing data
- [x] ✅ Committed preprocessing logic
- [x] ✅ Organized CSV dataset

---

## 📚 Learning Objectives

- Web scraping Google Play Store data
- Preprocessing real-world text data
- Structuring relational datasets for analysis
- Using Git for collaborative version control

---

## 🧠 Next Steps (Optional)

- 🔍 Sentiment and topic modeling using NLP
- 📈 Visualization in Matplotlib or Seaborn
- 🗃️ Store data in Oracle DB via SQLAlchemy or cx_Oracle
- 🧪 Add unit tests for scraper reliability

---

## 📬 Contact

**Gebrehiwot Tesfaye Assefa**  
📍 Addis Ababa, Ethiopia  
📧 tesfayegebrehiwot123@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/gebrehiwot-tesfaye-6646091a1/) | [Portfolio](https://portfolio-gebby.vercel.app/) | [GitHub](https://github.com/Gebrehiwot-Tesfaye)

---

## 📝 License

MIT License — feel free to use and adapt.
