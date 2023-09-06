#! /bin/bash
#reana-client secrets-add --file ~/cert/client.key --file ~/cert/client.crt --env #VOMSPROXY_PASS=cGFzc3dvcmQ= --env VONAME=escape --env RUCIO_USERNAME=msmirnov --overwrite

# for CTA dataset production 

export REANA_WORKON=cta_dataset
reana-client create -w $REANA_WORKON -f ./Analysis/reana/reana_cta_dataset.yml

# for KM3NeT dataset production

# export REANA_WORKON=km3net_dataset
# reana-client create -w $REANA_WORKON -f ./Analysis/reana/reana_km3net_dataset.yml

# for combined CTA and KM3NeT datasets production

# export REANA_WORKON=comb_dataset
# reana-client create -w $REANA_WORKON -f ./Analysis/reana/reana_comb_dataset.yml

reana-client upload
reana-client start
reana-client status
reana-client ls
