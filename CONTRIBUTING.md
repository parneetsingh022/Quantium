# Contributing to Quantium

🎉 Thanks for your interest in contributing to **Quantium**!  
Your help makes this project better, whether it’s fixing a bug, improving documentation, or adding new features.

---

## 🧩 Getting Started

1. **Fork** the repository on GitHub.  
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/quantium.git
   cd quantium
   ```
3. Create a new **branch** for your changes:
   ```bash
   git checkout -b docs/project-metadata
   ```
   *(Use a descriptive name, e.g. `fix/unit-division-bug`, `feature/numpy-support`, `docs/examples`)*

---

## ⚙️ Setting Up the Development Environment

1. Make sure you have **Python 3.9+** installed.
2. Install dependencies in editable mode:
   ```bash
   pip install -e .[dev]
   ```
3. Run tests to confirm everything works:
   ```bash
   pytest
   ```

---

## 🧠 Making Changes

- Keep code clean and consistent with **PEP 8**.
- Add **docstrings** and comments where helpful.
- Update or add **unit tests** for new functionality.
- If you’re fixing a bug, include a **test that fails** before your fix and **passes** after.

---

## 🧾 Writing Commit Messages

Use clear, concise messages that explain **what** and **why**, not just **how**.

**Examples:**
- `fix: correct unit division output`
- `feat: add numpy array support for Quantity objects`
- `docs: update changelog for 0.0.1a0 release`

---

## 🧱 Pull Requests

When your changes are ready:
1. Push your branch:
   ```bash
   git push origin your-branch-name
   ```
2. Open a **Pull Request** to the `main` branch.
3. In your PR description:
   - Explain *what* the change does.
   - Link any related issues.
   - Mention if it affects backward compatibility.

We’ll review your PR, suggest improvements if needed, and merge it once approved. 💪

---

## 🧭 Code of Conduct

Please be kind and respectful to others.  
This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

---

## 💡 Need Help?

If you have questions, feel free to:
- Open a **discussion** or **issue** on GitHub.
- Share suggestions or feedback — all ideas are welcome!

---

**Thank you for helping make Quantium better!**
