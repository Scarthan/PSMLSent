if (test-path .\nuget.exe){ 
   
}else{
    Invoke-WebRequest "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile "nuget.exe"
}

if (test-path Microsoft.ML.1.0.0-preview) {
    
}else {
    ./nuget.exe install Microsoft.ML -version 1.0.0-preview    
}

if (test-path bin) {
    
}else{
    new-item bin -ItemType Directory | out-null
    
}

if (test-path .\bin\*.dll) {
    
}else {
    Get-ChildItem "*\lib\netstandard*\*.dll" | copy-item -Destination ".\bin"
}

$url = "https://raw.githubusercontent.com/Scarthan/PSMLSent/main/yelp_labelled.txt"
if (test-path .\yelp_labelled.txt) {
    
}else {
    Invoke-WebRequest -Uri $url -OutFile "yelp_labelled.txt"    
}

if(test-path .\bin\CpuMathNative.dll){

}else {

    if ([Environment]::Is64BitProcess) {
        copy-item "$pwd\\Microsoft.ML.CpuMath.1.0.0-preview\runtimes\win-x64\native\CpuMathNative.dll" -Destination ".\bin"
    }else {
        copy-item "$pwd\\Microsoft.ML.CpuMath.1.0.0-preview\runtimes\win-x86\native\CpuMathNative.dll" -Destination ".\bin"
    }

}

Get-ChildItem "$pwd\bin\*.dll" | Where-Object {$_.Name -ne "CpuMathNative.dll"} | ForEach-Object {
    Add-Type -Path $_.FullName
}

$dataPath = "$pwd\yelp_labelled.txt"

$mlCOntext = [Microsoft.ML.MLContext]::new()

$columns = [System.Collections.Generic.List``1[Microsoft.ML.Data.TextLoader+Column]]::new()

$columns.Add([Microsoft.ML.Data.TextLoader+Column]::new("SentimentText", "String", 0))
$columns.Add([Microsoft.ML.Data.TextLoader+Column]::new("Label", "Boolean", 1))

$columns.Add([Microsoft.ML.Data.TextLoader+Column]::new("PredictedLabel", "Boolean", 2))
$columns.Add([Microsoft.ML.Data.TextLoader+Column]::new("Probability", "Single", 3))
$columns.Add([Microsoft.ML.Data.TextLoader+Column]::new("Score", "Single", 4))

$opt = [Microsoft.ML.Data.TextLoader+Options]::new()
$opt.Separators = "`t"
$opt.Columns = $columns
$opt.HasHeader = $false

$dataView = [Microsoft.ML.TextLoaderSaverCatalog]::LoadFromTextFile($mlCOntext.Data, $dataPath, $opt)

$splitDataView =  $mlCOntext.Data.TrainTestSplit($dataView, 0.2)
$trainSet = $splitDataView.TrainSet
$testSet = $splitDataView.TestSet

$estimator = [Microsoft.ML.TextCatalog]::FeaturizeText($mlCOntext.Transforms.Text, "Features", "SentimentText")

$optTrain = [Microsoft.ML.Trainers.SdcaLogisticRegressionBinaryTrainer+Options]::new()
$optTrain.FeatureColumnName = "Features"
$optTrain.LabelColumnName = "Label"

$optTrain.Shuffle = $false  

$trainer = [Microsoft.ML.StandardTrainersCatalog]::SdcaLogisticRegression($mlCOntext.BinaryClassification.Trainers, $optTrain)

$pipe = [Microsoft.ML.LearningPipelineExtensions]::Append($estimator, $trainer, "Everything")

$model = $pipe.Fit($trainSet)

$predict = $model.Transform($TestSet)

$mlCOntext.BinaryClassification.Evaluate($predict, "Label")