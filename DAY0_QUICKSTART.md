# Lab Dojo v8 - Ready to Use!

## ✅ Your Serverless Endpoint is LIVE

Your Vast.ai serverless is already configured and ready:

| Setting | Value |
|---------|-------|
| **Endpoint Name** | labdojo-qwen32b |
| **Endpoint ID** | 11809 |
| **Workergroup ID** | 16559 |
| **GPU** | RTX 5090 |
| **Cost** | $0.269/hour (only when active) |
| **Idle Cost** | $0.00/hour |

## 🚀 Start Lab Dojo Now

### Mac
```bash
# Double-click LabDojo_Installer.command
# OR in Terminal:
./LabDojo_Installer.command
```

### Windows
```
Double-click LabDojo_Installer.bat
```

**Dashboard opens at:** http://localhost:8080

## 🎯 What Happens When You Chat

### Simple Questions → Local (FREE)
- "What is 2+2?"
- "Capital of France?"
- Quick lookups

**Response time:** 2-5 seconds
**Cost:** $0.00

### Complex Questions → Serverless (Pay-per-use)
- "Analyze this research paper..."
- "Compare these methodologies..."
- Long reasoning tasks

**First request:** 60-90 seconds (cold start - workers spin up)
**After that:** 3-5 seconds
**Cost:** ~$0.0003 per request

## 💰 Cost Control

| Setting | Value |
|---------|-------|
| Daily Budget | $5.00 |
| When Exceeded | Routes to local only |
| Reset | Midnight daily |

**Typical daily cost:** $0.02 - $0.10

## 📊 Your Serverless Settings (Optimal for Low Cost)

```
Minimum Workers: 0      ← No idle charges!
Max Workers: 2          ← Enough for your usage
Coldstart Ratio: 2x     ← Fast scale-down
Target Utilization: 0.8 ← Efficient scaling
```

## 🔧 Files Included

| File | Purpose |
|------|---------|
| `labdojo.py` | Main application |
| `LabDojo_Installer.command` | Mac installer |
| `LabDojo_Installer.bat` | Windows installer |
| `docker/` | Serverless container (already deployed!) |

## ❓ FAQ

### Q: Do I need to set up anything else?
**A:** No! Your serverless endpoint is already live. Just run the installer.

### Q: Why does the first request take so long?
**A:** Cold start - workers spin up from 0. After that, it's fast (3-5 seconds).

### Q: How do I check my spending?
**A:** Dashboard shows "Today's Cost" in real-time.

### Q: What if I exceed the daily budget?
**A:** All requests automatically route to local Ollama (free).

## 🎉 You're Ready!

1. Run the installer
2. Open http://localhost:8080
3. Start chatting!

**Enjoy Lab Dojo!** 🧪
