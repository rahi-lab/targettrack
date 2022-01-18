import subprocess

class SubProcManager():
    """
    This class manages many subprocess launched from the GUI
    """
    def __init__(self):
        self.runnings={}
        self.status={}
        self.logs={}

    def run(self,key,arg,logfn):
        """
        Runs a subprocess.
        key: str: unique key for the launched subprocess
        arg: str: command to be ran
        logfn: str: the logfile reporting the process. The format should be undestood by check

        Return: (bool,str) representing (success,message)
        """
        self.check()
        if key in self.runnings.keys():
            return False,"Bug: cfpark00@gmail.com"
        try:
            self.logs[key]=logfn
            self.runnings[key]=subprocess.Popen(arg)#,stdout=subprocess.PIPE)
            return True,""
        except:
            return False, "bug: cfpark00@gmail.com"

    def check(self):
        """
        Checks the status of a subprocess, including its progress
        Return: the status of every launched subprocess, should be understood by QtPullNN.py
        """
        for key,process in self.runnings.items():
            ret=process.poll()
            try:
                f=open(self.logs[key],"r")
                self.status[key]=[el.split("=") for el in f.readlines()[-1].strip().split(" ")]
                f.close()
                if ret is not None:
                    if ret==0:
                        self.status[key].append(["Status","Success"])
                    else:
                        self.status[key].append(["Status","Failed"])
                else:
                    self.status[key].append(["PID",str(process.pid)])
            except:
                self.status[key]=[["Initializing","..."]]
        return self.status

    def free(self,key):
        """
        Free the key of a ran subprocess
        """
        del self.runnings[key]
        del self.status[key]
        del self.logs[key]

    def close(self,arg,msg):
        """
        Recursive close function from the GUI
        """
        ####Dependency
        ok=True#no dependencies
        ####Self Behavior
        if not ok:
            return False,msg
        if arg=="force":
            for key,process in self.runnings.items():
                msg+="Subprocess: Terminating "+key+" with pid "+str(process.pid)+"\n"
                process.terminate()
            return True,msg
        elif arg=="save":
            if len(self.runnings)==0:
                return True,msg
            msg+="Subprocess: We currently do not support safe-save of running subprocs.\n"
            return False,msg
