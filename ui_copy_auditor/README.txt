Run:
1) pip install -r requirements.txt
2) Put screenshots in a folder, e.g. screenshots/
3) Optional: start local LanguageTool at localhost:8081
   java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081 --allow-origin
4) Optional: start Ollama and pull a model, e.g. llama3.1:8b
5) Run:
   python ui_copy_auditor.py --input .\screen_shot --rules term_rules.yaml --output output
   .\.venv\Scripts\python.exe ui_copy_auditor.py --input .\screen_shot --output .\output --disable-ollama

Useful flags:
--disable-languagetool
--disable-ollama
--ollama-model llama3.1:8b

Run:
   streamlit run app.py
   .\.venv\Scripts\python.exe -m streamlit run app.py
