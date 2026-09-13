// =========================
// QISKIT
// =========================

async function loadQiskit() {
    try {
        const response = await fetch("data/qiskit.json");
        const data = await response.json();

        const container = document.getElementById("qiskit-container");

        container.innerHTML = "";

        data.forEach(item => {
            const card = document.createElement("article");

            card.className = "qiskit-card";

            card.innerHTML = `
                <div class="card-top">
                    <span class="card-number">${item.number}</span>
                    <span class="badge badge-open">${item.status}</span>
                </div>

                <div class="card-content">
                    <p class="card-category">
                        QISKIT STUDY
                    </p>

                    <h3>${item.title}</h3>

                    <p>${item.description}</p>

                    <div class="result-box">
                        ${item.result}
                    </div>
                </div>

                <div class="tag-list">
                    ${item.tags.map(tag => `<span>${tag}</span>`).join("")}
                </div>
            `;

            container.appendChild(card);
        });

    } catch (error) {
        console.error("Qiskit 데이터 로딩 실패:", error);
    }
}


// =========================
// SCHEDULE
// =========================

async function loadSchedule() {
    try {
        const response = await fetch("data/schedule.json");
        const data = await response.json();

        const container = document.getElementById("schedule-container");

        container.innerHTML = "";

        data.forEach(item => {
            const schedule = document.createElement("article");

            schedule.className = "schedule-item";

            schedule.innerHTML = `
                <div class="schedule-date">
                    ${item.date}
                </div>

                <div class="schedule-content">
                    <span class="schedule-type">
                        ${item.type}
                    </span>

                    <h3>
                        ${item.title}
                    </h3>

                    <p>
                        ${item.description}
                    </p>
                </div>
            `;

            container.appendChild(schedule);
        });

    } catch (error) {
        console.error("Schedule 데이터 로딩 실패:", error);
    }
}


loadQiskit();
loadSchedule();