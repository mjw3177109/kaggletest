# V2.9.19 Kaggle ERROR — authenticated read-only diagnostic

- Original Notebook: mjw3177109/kas-v2919-tabm-route-t4-s6e10
- Stage: kernel_failed
- Action: ONLY status and output reads. No kernel creation and NO Kaggle submission.

## Kaggle API command output
- Return code: 0
- Command: kaggle kernels status mjw3177109/kas-v2919-tabm-route-t4-s6e10
- Output: mjw3177109/kas-v2919-tabm-route-t4-s6e10 has status "KernelWorkerStatus.ERROR" |  | 
- Return code: 0
- Command: kaggle kernels output mjw3177109/kas-v2919-tabm-route-t4-s6e10 -p /tmp/kas_v2919_diagnostic -o -q
- Output:  | 

## Error and log files
### kas-v2919-tabm-route-t4-s6e10.log
~~~
[{"stream_name":"stderr","time":4.792144638,"data":"Traceback (most recent call last):\n"}
,{"stream_name":"stderr","time":4.792316044,"data":"  File \"/kaggle/src/script.py\", line 20, in \u003cmodule\u003e\n"}
,{"stream_name":"stderr","time":4.79233912,"data":"    import feature_engineering as route\n"}
,{"stream_name":"stderr","time":4.792343532,"data":"ModuleNotFoundError: No module named 'feature_engineering'\n"}
,{"stream_name":"stderr","time":4.792418738,"data":"\n"}
,{"stream_name":"stderr","time":4.792425337,"data":"During handling of the above exception, another exception occurred:\n"}
,{"stream_name":"stderr","time":4.7924292059999996,"data":"\n"}
,{"stream_name":"stderr","time":4.792432521,"data":"Traceback (most recent call last):\n"}
,{"stream_name":"stderr","time":4.792590659,"data":"  File \"/kaggle/src/script.py\", line 22, in \u003cmodule\u003e\n"}
,{"stream_name":"stderr","time":4.792600687,"data":"    import v2918_route_profile_lgbm_train as route\n"}
,{"stream_name":"stderr","time":4.792604869,"data":"ModuleNotFoundError: No module named 'v2918_route_profile_lgbm_train'\n"}
,{"stream_name":"stderr","time":7.309889487,"data":"/usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\\|'\n"}
,{"stream_name":"stderr","time":7.309942927,"data":"  cells[i][c] = re.sub('\\\\\\\\\\|', '|', cell)\n"}
,{"stream_name":"stderr","time":7.654018037,"data":"/usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\\_'\n"}
,{"stream_name":"stderr","time":7.654040053,"data":"  text = re.sub(r'_', '\\_', text) # Escape underscores in display text\n"}
,{"stream_name":"stderr","time":9.548753363,"data":"[NbConvertApp] Converting notebook __script__.ipynb to html\n"}
,{"stream_name":"stderr","time":10.694121017,"data":"[NbConvertApp] Writing 311159 bytes to __results__.html\n"}
]
~~~
