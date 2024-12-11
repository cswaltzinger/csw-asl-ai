all: env-create emb model test int 

RUN_CMD=conda run --no-capture-output -n csw-tz-img-final 
CMD_END=2> /dev/null

env-create: 
	@clear
	@echo  "=====>   Creating the environment   <====="	
	@echo  ""
	@echo  ""
	conda env create -f env.yaml

env-rm: 
	@clear 
	@echo  "=====>   Removing the environment   <====="	
	@echo  ""
	@echo  ""
	conda env remove -n csw-tz-img-final

emb: env-create
	@clear 
	@echo  "=====>   Generating the embeddings   <====="	
	@echo  ""
	@echo  ""
	$(RUN_CMD) python genEmb.py $(CMD_END)

model: env-create
	@clear
	@echo  "=====>   Making the model   <====="	
	@echo  ""
	@echo  ""
	$(RUN_CMD) python make_model.py $(CMD_END)


test: env-create
	@clear
	@echo  "=====>   Testing the model   <====="	
	@echo  ""
	@echo  ""
	$(RUN_CMD) python test.py $(CMD_END)

int: interactive
interactive: #env-create
	@clear
	@echo  "=====>   Interactive Mode (press Q to quit)   <====="	
	@echo  ""
	@echo  ""
	@$(RUN_CMD) python interactive.py $(CMD_END)


clean:
	@clear
	rm -rf *.pkl *.h5 __pycache__

deep-clean: clean env-rm

	