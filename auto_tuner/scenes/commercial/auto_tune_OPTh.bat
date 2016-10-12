@echo off
setlocal
python ../../AutoTuner.py pipeline_config=../../pipeline.xml scene_config=scene_OPTh.xml %*
endlocal