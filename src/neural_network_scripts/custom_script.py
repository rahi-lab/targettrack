######
#This is a template for a custom script to be ran.
#
######
import h5py

sys.path.append("src/neural_network_scripts/models")
st=time.time()
assert len(sys.argv)==3
## Don't change
logfn=sys.argv[2]
dataset_path=sys.argv[1]
dataset_name=os.path.split(dataset_path)[1].split(".")[0]
props=dataset_name.split("_")
NetName=props[-2]
runname=props[-1]
identifier="net/"+NetName+"_"+runname
