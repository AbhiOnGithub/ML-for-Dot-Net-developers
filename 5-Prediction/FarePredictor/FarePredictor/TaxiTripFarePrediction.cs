using System;
using Microsoft.ML.Runtime.Api;

namespace FarePredictor
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
