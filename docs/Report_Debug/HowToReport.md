 
-----

# Bug Reporting Guide

Thank you for taking the time to help make ShieldX better\! A well-written, detailed bug report helps us understand, reproduce, and fix the issue as quickly as possible.

Before creating a new report, please **[search the existing issues](https://www.google.com/search?q=https-link-to-your-issues-page)** to ensure the bug has not already been reported.

## Reporting Process

We encourage a 2-step process to ensure the quality of every report.

### Step 1: Prepare Your Report (Draft Locally)

Before submitting a report on GitHub, please take a moment to gather all the necessary information. Drafting the report in a local file first is an excellent habit.

1.  **Identify the Service:** First, clearly determine which service in the system is affected by the bug. For example: `auth service`, `ingress service`, `guardian service`.

2.  **Create a Draft Markdown File:** To draft your report, you can create a local `.md` file. We suggest a naming convention to help you stay organized:
    `<service-name>-<short-bug-desc>-<github-id>.md`

      * **Example:** `auth-login-fails-shield_team.md`

3.  **Draft the Report:** Use the detailed template in Step 2 below to fill in all the necessary information in your local markdown file.

### Step 2: Fill Out and Submit the Report on GitHub

Once you have all the information ready in your draft file, navigate to the project's [**Issues**](https://www.google.com/search?q=https-link-to-your-issues-page) tab and create a new report.

Use the following template to ensure your report is complete and professional.

-----

### Bug Report Template

**Title:**
*(Use a concise and descriptive title with the following structure: `bug(<service-name>): <short-description-of-bug>`)*

  * **Example:** `bug(auth): Login fails when username contains special characters`

**Affected Service:**
*(Specify the name of the service or system component that has the bug.)*

  * **Example:** `auth-service`

**Bug Description:**
*(Provide a clear and concise summary of the issue you are experiencing.)*

**Steps to Reproduce:**
*(Provide a specific list of steps so we can reproduce the bug. This is the most important section.)*

1.  ...
2.  ...
3.  ...

**Expected Behavior:**
*(A brief description of what you expected to happen.)*

**Actual Behavior:**
*(A description of what actually happened. If there was an error message, please copy and paste it here inside a code block.)*

```
(Paste error messages or logs here)
```

**Environment:**
*(Please provide details about your environment to help us diagnose the issue.)*

  * **Operating System:** (e.g., Ubuntu 22.04, Windows 11, macOS Sonoma)
  * **Go Version:** (e.g., go version go1.22.1 linux/amd64)
  * **Docker Version:** (e.g., Docker version 24.0.5)

**Severity (Optional):**
*(Assess the impact of this bug. Please select one of the following levels.)*

  * [ ] **Critical:** System crash, data loss, or a major security vulnerability.
  * [ ] **High:** A major feature is non-functional with no available workaround.
  * [ ] **Medium:** A minor feature is non-functional, or a major feature behaves inconsistently.
  * [ ] **Low:** A cosmetic issue, typo, or minor UI problem that doesn't significantly affect functionality.

**Additional Context:**
*(Add any other information you think is helpful, such as screenshots, videos, or your own speculations about the cause of the bug.)*

-----

This process encourages contributors to prepare their reports thoroughly while leveraging the power of GitHub Issues to track and manage bugs effectively.