# PayLens — Payment Network Intelligence

A synthetic payment analytics platform designed to answer business questions across customers, merchants, regions and transaction categories.

## Questions answered
- Which merchant categories are growing fastest?
- Which regions have the highest transaction volume?
- Which customers are high-value?
- Where are unusual transaction patterns emerging?
- How do payment behaviors differ by segment?

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py
streamlit run dashboard/app.py
```
