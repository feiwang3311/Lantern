source /opt/intel/vtune_amplifier_xe/amplxe-vars.sh

amplxe-cl -collect advanced-hotspots -- ./train.sh

# amplxe-gui
