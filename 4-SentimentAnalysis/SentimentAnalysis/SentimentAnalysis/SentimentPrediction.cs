using System;
using Microsoft.ML.Runtime.Api;

namespace SentimentAnalysis
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}
