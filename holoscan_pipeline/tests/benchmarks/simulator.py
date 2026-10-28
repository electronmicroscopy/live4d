"""Drive only the isolated live4d-integration-sim simulator, sample resources, and report drops."""
import json,pathlib,re,subprocess,threading,time,urllib.request
base='http://127.0.0.1:18081'
log=pathlib.Path('/tmp/live4d-isolated-stress.log')
name='live4d-integration-isolated-stress'
def api(method,path,value=None):
    data=None if method=='GET' else (b'' if value is None else json.dumps({'value':value}).encode())
    req=urllib.request.Request(base+path,data=data,method=method,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=75) as response:
        body=response.read();return json.loads(body) if body else None
for _ in range(400):
    if '[holoscan] ready' in log.read_text(): break
    time.sleep(.1)
else: raise RuntimeError('receiver not ready')
api('PUT','/detector/api/1.8.0/config/nimages',128)
print('warmup_arm',api('PUT','/detector/api/1.8.0/command/arm'),flush=True)
api('PUT','/detector/api/1.8.0/command/trigger')
for _ in range(400):
    if 'parallax_recon_scan001.npy' in log.read_text(): break
    time.sleep(.05)
else: raise RuntimeError('warmup did not complete')
print('warmup_dropped',api('GET','/stream/api/1.8.0/status/dropped'),flush=True)
api('PUT','/detector/api/1.8.0/config/nimages',3600000)
print('arm',api('PUT','/detector/api/1.8.0/command/arm'),flush=True)
started=time.monotonic();source_start=time.time();print('source_start',source_start,flush=True)
def trigger():
    print('trigger',api('PUT','/detector/api/1.8.0/command/trigger'),'elapsed',time.monotonic()-started,flush=True)
threading.Thread(target=trigger,daemon=True).start()
records=[]
while time.monotonic()-started < 100:
    stats=subprocess.run(['docker','stats','--no-stream','--format','{{json .}}',name],capture_output=True,text=True).stdout.strip()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    text=log.read_text();matches=re.findall(r'parallax branch batches=\d+ frames=(\d+)',text)
    record=dict(elapsed=time.monotonic()-started,stats=stats,gpu=gpu,parallax_frames_lower_bound=int(matches[-1]) if matches else 0)
    records.append(record);print(json.dumps(record),flush=True)
    if '[benchmark-complete]' in text:
        complete=json.loads(text.split('[benchmark-complete] ')[-1].splitlines()[0]);print('completed_wall_seconds',complete['time']-source_start,flush=True);break
    time.sleep(1)
print('after',api('GET','/stream/api/1.8.0/status/dropped'),flush=True)
pathlib.Path('/tmp/live4d-isolated-stress-resources.json').write_text(json.dumps(records,indent=2))
