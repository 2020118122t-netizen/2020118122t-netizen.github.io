// ===============================
// QISKIT DATA LOAD
// ===============================

async function loadQiskit() {

    const container =
        document.getElementById("qiskit-container");

    try {

        const response =
            await fetch("./data/qiskit.json");

        if (!response.ok) {
            throw new Error(
                "qiskit.json 파일을 불러오지 못했습니다."
            );
        }

        const data =
            await response.json();


        container.innerHTML = "";


        if (data.length === 0) {

            container.innerHTML = `
                <p class="loading-text">
                    아직 등록된 Qiskit 데이터가 없습니다.
                </p>
            `;

            return;
        }


        data.forEach((item, index) => {

            const element =
                document.createElement(
                    item.link ? "a" : "article"
                );


            if (item.link) {
                element.href = item.link;
            }


            element.className =
                index === 0
                ? "qiskit-card qiskit-card-active"
                : "qiskit-card";


            const tags =
                (item.tags || [])
                .map(tag => `<span>${tag}</span>`)
                .join("");


            const resultHTML =
                item.result
                ? `
                    <div class="result-box">
                        ${item.result}
                    </div>
                  `
                : "";


            const linkHTML =
                item.link
                ? `
                    <div class="card-link">
                        View Assignment →
                    </div>
                  `
                : "";


            element.innerHTML = `

                <div class="card-top">

                    <span class="card-number">
                        ${item.number || ""}
                    </span>

                    <span class="badge badge-open">
                        ${item.status || "OPEN"}
                    </span>

                </div>


                <div class="card-content">

                    <p class="card-category">
                        ${item.category || "QISKIT STUDY"}
                    </p>

                    <h3>
                        ${item.title}
                    </h3>

                    <p class="card-description">
                        ${item.description || ""}
                    </p>

                    ${resultHTML}

                </div>


                <div class="tag-list">
                    ${tags}
                </div>

                ${linkHTML}
            `;


            container.appendChild(element);

        });


    } catch (error) {

        console.error(error);

        container.innerHTML = `
            <div class="error-box">
                Qiskit 데이터를 불러오지 못했습니다.
                data/qiskit.json 파일을 확인해주세요.
            </div>
        `;
    }
}



// ===============================
// SCHEDULE DATA LOAD
// ===============================

async function loadSchedule() {

    const container =
        document.getElementById(
            "schedule-container"
        );


    try {

        const response =
            await fetch(
                "./data/schedule.json"
            );


        if (!response.ok) {

            throw new Error(
                "schedule.json 파일을 불러오지 못했습니다."
            );

        }


        const data =
            await response.json();


        container.innerHTML = "";


        if (data.length === 0) {

            container.innerHTML = `
                <p class="loading-text">
                    아직 등록된 일정이 없습니다.
                </p>
            `;

            return;
        }


        data.forEach(item => {

            const element =
                document.createElement(
                    "article"
                );


            element.className =
                "schedule-item";


            element.innerHTML = `

                <div class="schedule-date">
                    ${item.date}
                </div>


                <div class="schedule-content">

                    <span class="schedule-type">
                        ${item.type || "PLAN"}
                    </span>

                    <h3>
                        ${item.title}
                    </h3>

                    <p>
                        ${item.description || ""}
                    </p>

                </div>
            `;


            container.appendChild(element);

        });


    } catch (error) {

        console.error(error);

        container.innerHTML = `
            <div class="error-box">
                Schedule 데이터를 불러오지 못했습니다.
                data/schedule.json 파일을 확인해주세요.
            </div>
        `;

    }

}



// ===============================
// PAGE START
// ===============================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadQiskit();
        loadSchedule();

    }
);