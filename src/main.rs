use nn::{NN, HaltCondition};

fn main() {
    // Create Examples of XOR Function
    // First Vector = Inputs
    // Second Vector = Expected Outputs
    let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];

    // Input Layer = 2 Nodes
    // One Hidden Layer = 3 Nodes
    // Output Layer = 1 Node
    let mut net = NN::new(&[2, 3, 1]);

    // Train
    net.train(&examples)
        .halt_condition(HaltCondition::Epochs(9000000))
        .log_interval(Some(100))
        .momentum(0.1)
        .rate(0.3)
        .go();

    // Evaluate
    for &(ref inputs, ref outputs) in examples.iter() {
        let results = net.run(inputs);
        let (result, key) = (results[0].round(), outputs[0]);
        assert!(result == key);
    }
}
