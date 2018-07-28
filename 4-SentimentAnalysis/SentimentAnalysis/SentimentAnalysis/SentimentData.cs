using System;
using Microsoft.ML.Runtime.Api;

namespace SentimentAnalysis
{
    public class SentimentData
    {
        [Column("0")]
        public string SentimentText;

        [Column("1", name: "Label")]
        public float Sentiment;
    }
}
