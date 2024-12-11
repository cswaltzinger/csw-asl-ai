# Image Processing Final Project
In order to make our project easy to implement and run, we created a makefile that can setup everything that is needed.  

## Quickstart
Simply go into the project directory and execute ```make```.  This command will create the necessary conda enviornment, generate the embeddings based on the raw photos, create/compile a neural network, train/save/test the created network, and run interactive mode.

If you wish to run the rules one at a time and not use the **all** rule, then simply execute the following rules sequentially: env-create, emb, model, test, int. 

### Makefile Rules: 
- **all:** this command executes the following rules sequentially:
    - env-create 
    - emb 
    - model 
    - test 
    - int 
- **env-create:** Creates the necessary conda env to run the programs.
- **env-rm:** Remove the created conda env.
- **emb:** Generate and pickle the necessary embedings to train the neural network.
- **model:** Create and train the nerual network.  
- **test:** Test the created nerual network with a random subset of the training data.  
- **int:** Alias rule for **interactive**
- **interactive:** Run interactive mode and use the model to interpret ASL in real time.  
- **clean:** remove all local pkl, h5, and \_\_pycache\_\_ files/directories.  
- **deep-clean:** remove all local pkl, h5,  \_\_pycache\_\_ files and remove the created conda env.


