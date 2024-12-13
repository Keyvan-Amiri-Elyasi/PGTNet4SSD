export DIRECTORY=conversion_configs
export CONFIG=helpdesk.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
export CONFIG=sepsis.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
export CONFIG=bpic13i.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
export CONFIG=bpic15m1.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
export CONFIG=bpic20d.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
export CONFIG=bpic20i.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
export CONFIG=bpic12.yaml
python GTconvertor.py ${DIRECTORY} ${CONFIG} --overwrite true --ssd
