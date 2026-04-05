---
description: How to commit and push code on this Windows PowerShell environment
---

## Critical CLI Rules for PowerShell

1. **NEVER use `&&` to chain commands in PowerShell.** Use `;` (semicolons) instead.
   - ❌ `git add . && git commit -m "msg" && git push`
   - ✅ `git add .; git commit -m "msg"; git push`

2. **NEVER use `grep` in PowerShell.** Use `Select-String` (alias `sls`) instead.
   - ❌ `pip freeze | grep langchain`
   - ✅ `pip freeze | sls langchain`

3. **When calling python with `-c` and the path has spaces, use `&` (call operator).**
   - ❌ `"C:\path with spaces\python.exe" -c "print('hi')"`
   - ✅ `& "C:\path with spaces\python.exe" -c "print('hi')"`

## Git Push Workflow

This project uses a **personal GitHub account** that is different from the system's default (office) Git credentials.

// turbo-all

1. Stage changes:
```powershell
git add .
```

2. Commit:
```powershell
git commit -m "your message"
```

3. Push using a personal access token (PAT) embedded in the remote URL:
```powershell
git push
```

> **NOTE:** If push fails with a 403 Permission error, the user likely needs to update the remote URL with their personal PAT:
> ```powershell
> git remote set-url origin https://<PERSONAL_PAT>@github.com/Aryan-coder-student/NeuroVision-BHPC-VQA.git
> ```
> The user must supply their own PAT. Generate one at: https://github.com/settings/tokens
