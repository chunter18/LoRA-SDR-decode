@echo off
echo === LoRa IQ Pipe ===
echo Tuning Pluto to 915 MHz, piping raw IQ to stdout
echo.
echo Usage: run_iq_server.bat ^| python chirp_waterfall.py
echo.
"C:\Program Files\PothosSDR\bin\rtl_433-rtlsdr-soapysdr.exe" ^
  -d "driver=plutosdr" -f 915M -s 1000000 -t "bandwidth=1000000" ^
  -R 0 -w -
