source /opt/intel/vtune_amplifier_xe/amplxe-vars.sh
source /opt/intel/vtune_amplifier_xe/sep_vars.sh
source /opt/intel/advisor/advixe-vars.sh

# amplxe-cl -collect advanced-hotspots -- ./train.sh
amplxe-cl -collect hotspots -- ./train.sh

# amplxe-cl -collect memory-access -- ./train.sh

# amplxe-cl -collect-with runsa -knob event-config=? -- ./train.sh
# amplxe-cl -collect general-exploration -- ./train.sh
# amplxe-gui
