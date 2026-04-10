@echo off
cd /d D:\LanguageTool
if not exist server.properties (
    type nul > server.properties
)
java -cp languagetool-server.jar org.languagetool.server.HTTPServer --config server.properties --port 8081 --allow-origin
pause