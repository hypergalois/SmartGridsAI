import mlrun

@mlrun.pipeline(name="Training pipeline", project="smartgrids")
def training_pipeline(file_path: str):
    
    # Preprocesar datos
    preprocess_run = mlrun.run_function('preprocess-data', params={'file_path': file_path})
    
    # Entrenar modelo
    train_run = mlrun.run_function('train-model', params={'file_path': preprocess_run.outputs['consumo_electrico']})
    
    # Evaluar modelo
    eval_run = mlrun.run_function('evaluate-model', params={
        'model_path': train_run.outputs['model_path'],
        'test_file': preprocess_run.outputs['consumo_electrico']
    })