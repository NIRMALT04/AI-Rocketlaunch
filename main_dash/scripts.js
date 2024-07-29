document.getElementById('pidButton').addEventListener('click', function() {
    const kp = parseFloat(document.getElementById('kp').value);
    const ki = parseFloat(document.getElementById('ki').value);
    const kd = parseFloat(document.getElementById('kd').value);
    const data = {
        kp: kp,
        ki: ki,
        kd: kd,
        setpoint: 100,
        measurement: 80
    };

    fetch('/pid', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('controlSignal').innerText = `Control signal: ${data.control_signal}`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

document.getElementById('trainModelButton').addEventListener('click', function() {
    fetch('/train_model')
    .then(response => response.json())
    .then(data => {
        document.getElementById('modelAccuracy').innerText = `Model accuracy: ${(data.accuracy * 100).toFixed(2)}%`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

document.getElementById('sendTelemetryButton').addEventListener('click', function() {
    const telemetryData = document.getElementById('telemetryData').value;
    const data = {
        telemetry: telemetryData
    };

    fetch('/send_telemetry', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('telemetryStatus').innerText = data.message;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

document.getElementById('getEnvironmentButton').addEventListener('click', function() {
    fetch('/rocket_environment')
    .then(response => response.json())
    .then(data => {
        document.getElementById('environmentData').innerText = `Wind speed: ${data.wind_speed.toFixed(2)} m/s, Wind direction: ${data.wind_direction.toFixed(2)} degrees`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
