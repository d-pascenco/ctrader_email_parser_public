Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "C:\cTrader_email\run.bat" & Chr(34), 0
Set WshShell = Nothing
