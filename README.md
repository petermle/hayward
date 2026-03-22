# 🎓 ProfessorMatch

> A multi-source professor recommendation system that helps university students make smarter course enrollment decisions.

---

## 📌 Overview

Choosing the right professor significantly affects a student's academic performance, workload balance, and career trajectory — yet most students rely on informal word-of-mouth or a single rating site. **ProfessorMatch** addresses this by fusing data from multiple sources (RateMyProfessors, LinkedIn, and official course information) into a unified, explainable recommendation engine. Students specify their personal goal — GPA optimization, deep learning, or career development — and receive ranked professor recommendations with transparent scoring breakdowns, potential concerns, and LinkedIn networking insights.

---

## 🎬 Demo Workflow

1. Open the app and set your **course filter** (school, department, course code)
2. Select your **primary objective** — GPA Focused, Balanced, Learning Focused, or Career Focused
3. Adjust **workload tolerance** and optional preference checkboxes
4. Click **Generate Recommendations**
5. Review your **top professor match** with fit score and data confidence
6. Browse **professor cards** showing why each professor was recommended and any concerns
7. Explore the **Compare Professors** tab for a side-by-side dimension breakdown
8. Check the **LinkedIn networking panel** for professors worth connecting with

---

## 🗂️ File Structure

```
professor-match/
├── app.py                              # Streamlit frontend — main demo UI
├── app_v1_baseline.py                  # Baseline UI snapshot (rollback-safe copy)
├── professor_recommendation_engine.py  # Core recommendation logic — scoring, fusion, ranking
├── schemas.py                          # All data schemas (RMP, LinkedIn, unified feature, output)
├── requirements.txt                    # Python dependencies
└── README.md
```

**Key design principle:** `app.py` only imports from `professor_recommendation_engine.py` and calls its functions — no recommendation logic lives in the frontend.

---

## ⚙️ Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Chenghui-Tan/professor-match.git
cd professor-match

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` and loads mock professor data automatically — no database or API key required.

To run the baseline UI snapshot:

```bash
streamlit run app_v1_baseline.py
```

---

## 👤 Example User Scenario

**Scenario:** Maya is a second-year Computer Science student registering for CS301. She wants to build practical skills for a software engineering role while keeping her GPA competitive.

1. She selects **Career Focused** as her primary objective
2. Checks **Career-oriented student** and **Open to LinkedIn networking**
3. ProfessorMatch ranks Prof. Bob Smith #1 — 88% fit score — citing his 12 years of industry experience, startup background, and active LinkedIn profile
4. The networking panel surfaces two connect reasons and links directly to his LinkedIn profile
5. A controversy warning flags mixed student feedback on workload, helping Maya set expectations before enrolling

---

## 🏗️ How the Recommendation Engine Works

The engine is fully **rule-based and explainable** — no machine learning.

| Step | What happens |
|------|-------------|
| 1. Feature extraction | Raw RMP, LinkedIn, and syllabus fields are normalized to 0–1 scores across 10 dimensions |
| 2. Multi-source fusion | Dimensions are blended using source-specific weights (RMP primary, LinkedIn secondary) |
| 3. Mode scoring | A weighted sum is computed under the chosen goal mode |
| 4. Preference nudges | Small ±3–5% adjustments applied for individual checkbox preferences |
| 5. Ranking & explanation | Professors sorted by final fit score; human-readable reasons generated from score thresholds |

### Data Sources

| Source | Signals | Confidence weight |
|--------|---------|------------------|
| RateMyProfessors | Teaching quality, difficulty, workload, clarity, fairness | 50% |
| LinkedIn | Career relevance, industry experience, academic strength, networking | 30% |
| Official course info | Syllabus structure, grading policy, exam format | 20% |

### Ranking Modes

| Mode | Prioritises |
|------|------------|
| 🎯 GPA Focused | Low difficulty, fair grading, low workload |
| ⚖️ Balanced | Quality, clarity, fairness, moderate workload |
| 📚 Learning Focused | Clarity, engagement, intellectual challenge |
| 💼 Career Focused | Industry experience, career relevance, networking potential |

---

## 🔮 Future Improvements

- **Live data scraping** — replace mock data with real-time RMP and LinkedIn scraper pipelines
- **Personalized history** — store past preferences and enrollment outcomes to refine nudge weights over time
- **Collaborative filtering layer** — incorporate patterns from students with similar academic profiles
- **Course-level filtering** — match on specific course codes and semesters, not just professor names
- **Sentiment analysis** — replace keyword heuristics with an NLP model for review text scoring
- **Grade distribution integration** — use official registrar data to ground-truth the fairness and difficulty signals
- **Mobile-friendly layout** — adapt the Streamlit UI for smaller screens

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.32 |
| Data manipulation | Pandas 2.2 |
| Recommendation logic | Plain Python — rule-based weighted scoring |
| Data schemas | Python typed dicts (no ORM) |
| Language | Python 3.12 |
| Dependency management | pip + virtualenv |

No external ML frameworks, no database, no API keys required to run the demo.

---

## 📄 License

Built for academic and portfolio purposes.
