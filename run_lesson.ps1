param(
    [Parameter(Position=0)]
    [ValidateSet('lesson2_corrected.py', 'lesson3_exercise.py', 'lesson3_solution.py', 'lesson4_exercise.py', 'lesson4_solution.py', 'lesson5_exercise.py', 'lesson5_solution.py', 'lesson6_exercise.py', 'lesson6_solution.py', 'lesson6_predict.py', 'train_heat_fno.py', 'predict_heat.py', 'burgers_data.py', 'train_burgers_fno.py', 'predict_burgers.py')]
    [string]$Lesson = 'lesson2_corrected.py',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$LessonArguments
)

$lessonPython = Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.14-64\python.exe'
if (-not (Test-Path -LiteralPath $lessonPython)) {
    $lessonPython = (Get-Command python -ErrorAction Stop).Source
}
& $lessonPython (Join-Path $PSScriptRoot $Lesson) @LessonArguments
exit $LASTEXITCODE
