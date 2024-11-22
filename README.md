## Steps 


##### Setup environment  

* `python -m venv customer-satisfaction-env`  
* `source customer-satisfaction-env/bin/activate`  
* `customer-satisfaction-env/bin/pip install zenml["server"]`  
* `customer-satisfaction-env/bin/pip install scikit-learn`  
* `zenml init`  
* `zenml up`  
    * This will launch zenml here `http://127.0.0.1:8237/login?username=default`      

#### Troubleshooting 

* If zenml crashes, check logs using `zenml logs -f`, if you see zenml server crash with below error :  
```
objc[31819]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.   
objc[31819]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called. We cannot safely call  
it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug. 
```   
Set this variable in your .bash_profile :  
`export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`  

