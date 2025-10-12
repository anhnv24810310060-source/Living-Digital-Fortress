 
-----

# 🗺️ ShieldX-Admin Roadmap

Welcome to the ShieldX development roadmap\! This document aims to provide transparency on our priorities and guide the future direction of the project. ShieldX is currently in an **ALPHA** stage, which means we are actively developing core features and major changes are possible.

This roadmap is a living document and will be updated regularly based on community feedback and emerging requirements.

### Legend

  * ✅ **Done**
  * 🚧 **In Progress**
  * 💡 **Planned / Idea**
  * 🤔 **Under Discussion**

### 👋 How to Contribute to the Roadmap

Your input is invaluable\! If you have an idea for a new feature or would like to propose a change, please:

1.  **Discuss Big Ideas:** Open a new topic in [**GitHub Discussions**](https://github.com/shieldx-bot/shieldx/discussions) for the community to discuss together.
2.  **Request Specific Features:** Create a [**New Issue**](https://www.google.com/search?q=https://github.com/shieldx-bot/shieldx/issues/new/choose) using the "Feature Request" template.

-----





## 🚀 Phase 1: The Foundation & Core Visibility  

*Goal: Build the Minimum Viable Product (MVP) of the Dashboard, providing the most critical information on system health and security events.*

  - **A[ ] 💡 Establish `shieldx-admin` Service**

      - [✅ ] 💡 Create the basic project structure for the dashboard service 
         packages : 
           
          ```
          

           ```
          --Frontend-- 
          React : 19.1.0 
          React-redux: 9.2.0
          "socket.io-client": 4.8.1,
          Typescript: 5.9.3
          react-dom: 19.1.0
          tailwindcss/postcss: 4
          
          ```

      - [ ] 💡 Build a secure authentication mechanism for administrators. (After All)

  - **[ ] 💡 Main Dashboard Page**

      - [✅] 💡 **Key KPI Metrics Widget:** Display critical stats for the last 24 hours (Total Requests, Threats Blocked, Avg. Latency....
      - [✅] 💡 **System Health Status Widget:** Show the status (Online/Offline/Warning) of each microservice (Orchestrator, Guardian, Credits...).
      - [🤔] 💡 **Latest Alerts Widget:** Display the 10 most recent high-severity security alerts.

  - **[ ] 💡 Event Investigation Page**

      - [ ] 💡 Build the UI for real-time viewing of security event logs.
      - [ ] 💡 Provide basic filters: by time range, severity level, and attack type.

## 🚶 Phase 2: Actionable Insights & Control  

*Goal: Expand the Dashboard's capabilities from just "viewing" to "analyzing" and "taking action".*

  - **[ ] 💡 Main Dashboard Enhancements**

      - [ ] 💡 **Live Attack Map:** Integrate a world map visualizing the geographic origin of attacks.
      - [ ] 💡 **Threat Trend Chart:** Add a line chart showing attack volume over time.
      - [ ] 💡 **Top 5 Attack Vectors Widget:** Display the most frequently attacked IPs and API endpoints.

  - **[ ] 💡 Policy Management**

      - [ ] 💡 Build a **read-only** interface to view the content of currently applied OPA policies.
      - [ ] 🤔 **Policy Editor:** Allow creating and editing OPA policies directly from the UI.

  - **[ ] 💡 Deeper Service Integrations**

      - [ ] 💡 **Event Details View:** Click an alert to see the full request details (headers, body...) and the reason it was blocked.
      - [ ] 💡 **User Behavior Analytics Page:** A basic interface to view a user's risk score and suspicious activities (integrates with `ContAuth`).

## 🏃 Phase 3: Intelligence & Automation  

*Goal: Introduce intelligent (AI) and automation elements to the Dashboard to reduce the administrative workload.*

  - **[ ] 💡 Reporting & Analytics**

      - [ ] 💡 Allow exporting event data from the Investigation page to a CSV file.
      - [ ] 💡 Automatically generate and email a weekly security summary report.

  - **[ ] 💡 Full Service Integration**

      - [ ] 💡 **Sandbox Analysis Reports:** An interface to view detailed reports from `Guardian` on analyzed malicious payloads.
      - [ ] 💡 **Quick Actions:** Add "Block IP" or "Isolate User" buttons directly within the alert investigation view.

  - **[ ] 💡 Intelligent Features**

      - [ ] 🤔 **AI-Powered Policy Suggestions:** The system automatically analyzes attack patterns and suggests new rules/policies.
      - [ ] 🤔 **"Attack Playback":** A graphical interface that reconstructs the event sequence of a complex attack over a timeline.

## 🌌 Future Ideas

*Goal: Big-picture ideas to establish ShieldX as a leading security platform.*

  - [ ] 🤔 Build specialized dashboards for compliance standards (e.g., PCI-DSS, GDPR).
  - [ ] 🤔 Integrate with third-party platforms like Slack (for alerts), JIRA (for ticketing), and SIEMs.
  - [ ] 🤔 A user-facing UI for managing billing and resource limits (integrates with `Credits Service`).

-----

*Note: This roadmap is subject to change. Priorities may be adjusted based on user needs and community contributions.*