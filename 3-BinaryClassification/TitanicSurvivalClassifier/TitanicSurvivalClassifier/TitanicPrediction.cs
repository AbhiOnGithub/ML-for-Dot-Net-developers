using System;
using Microsoft.ML.Runtime.Api;

namespace TitanicSurvivalClassifier
{
    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Survived;
    }
}
