export DIRECTORY=conversion_configs
export CONFIG=helpdesk.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true 
export CONFIG=sepsis.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=bpic15m1.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=bpic15m2.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=bpic15m3.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=bpic15m4.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=bpic15m5.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=bpic12.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true
export CONFIG=hospital.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true