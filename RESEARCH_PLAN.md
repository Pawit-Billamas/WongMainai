# Wongnai Thai Restaurant Reviews — Aspect Sentiment, Complaint Clustering & Mixed-Review NLP Project

**Dataset:** [Wongnai/wongnai_reviews](https://huggingface.co/datasets/Wongnai/wongnai_reviews) (Thai restaurant reviews + 1–5 star ratings)

---

## Research Questions (RQs)

1. **RQ1 (Overall vs. Mixed-Review Trap):**  
   On reviews with both praise and complaints (*"อาหารอร่อย แต่รอนานมาก บริการแย่"*), how often does a single overall rating/sentiment fail compared to **aspect-level extraction** (`อาหาร`, `บริการ`, `ราคา`, `บรรยากาศ/ความสะอาด`)?
2. **RQ2 (Complaint Clustering / Grouping):**  
   When we extract negative review segments and cluster them using `TF-IDF` vs. `multilingual-e5-base` (+ K-Means), what are the main **customer pain-point clusters** (e.g., *รอนาน*, *พนักงานบริการ*, *รสชาติ/ความสด*, *ราคา/ปริมาณ*), and which model produces cleaner clusters?
3. **RQ3 (Indirect / Implicit Complaints):**  
   When customers complain **without naming the aspect word directly** (e.g., *"สั่งน้ำไป 45 นาทีแล้วยังไม่ได้เลย"* instead of saying *"บริการช้า"*), can Keyword rules, `multilingual-e5-base`, and `Gemma 4 12B` still identify the right aspect (`บริการ`)?

---

## Project Structure

- `RESEARCH_PLAN.md` — Project design, RQs, and 5-phase plan
- `data/` — Sampled & annotated subsets from `Wongnai/wongnai_reviews`
- `notebooks/` — Clean, reproducible experiment notebooks (ready to run step-by-step)
- `results/` — Tables, clusters, and charts
