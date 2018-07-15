using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace ClusteringInML
{
    public static class Program
    {

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris-data.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        private static void Main(string[] args)
        {

            PredictionModel<IrisData, ClusterPrediction> model = Train();

            model.WriteAsync(_modelPath);

            var prediction = model.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");

        }

        private static PredictionModel<IrisData, ClusterPrediction> Train()
        {

            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(_dataPath).CreateFrom<IrisData>(separator: ','));

            pipeline.Add(new ColumnConcatenator(
                    "Features",
                    "SepalLength",
                    "SepalWidth",
                    "PetalLength",
                    "PetalWidth"));

            pipeline.Add(new KMeansPlusPlusClusterer() { K = 3 });

            var model = pipeline.Train<IrisData, ClusterPrediction>();
            return model;
        }
    }
}
