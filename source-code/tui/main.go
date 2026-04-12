package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/wrap"
)

// ─────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────

const (
	apiBase    = "http://localhost:8000"
	appVersion = "2.0.0"
)

// ─────────────────────────────────────────────────────────────────
// Colours
// ─────────────────────────────────────────────────────────────────

var (
	cBgBase     = lipgloss.Color("#1a1916")
	cBgSurface  = lipgloss.Color("#211f1c")
	cBgElevated = lipgloss.Color("#2a2825")
	cBgHover    = lipgloss.Color("#312f2b")
	cBorderSub  = lipgloss.Color("#2d2b28")
	cAccent     = lipgloss.Color("#d97706")
	cAccentDim  = lipgloss.Color("#92400e")
	cTextPri    = lipgloss.Color("#f5f0e8")
	cTextSec    = lipgloss.Color("#a89880")
	cTextMuted  = lipgloss.Color("#6b6057")
	cTextAmber  = lipgloss.Color("#f59e0b")
	cUserFg     = lipgloss.Color("#fef3c7")
	cGreen      = lipgloss.Color("#4ade80")
	cRed        = lipgloss.Color("#f87171")
)

// ─────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────

var (
	sBgFull    = lipgloss.NewStyle().Background(cBgBase)
	sDivider   = lipgloss.NewStyle().Foreground(cBorderSub)
	sMuted   = lipgloss.NewStyle().Foreground(cTextMuted)
	sPrimary = lipgloss.NewStyle().Foreground(cTextPri)
	sAccent    = lipgloss.NewStyle().Foreground(cAccent)
	sAccentB   = lipgloss.NewStyle().Foreground(cAccent).Bold(true)
	sError     = lipgloss.NewStyle().Foreground(cRed)

	sUserLabel = lipgloss.NewStyle().Foreground(cAccent).Bold(true)
	sAiLabel   = lipgloss.NewStyle().Foreground(cTextAmber).Bold(true)
	sTimestamp = lipgloss.NewStyle().Foreground(cTextMuted)
	sUserText  = lipgloss.NewStyle().Foreground(cUserFg)
	sAiText    = lipgloss.NewStyle().Foreground(cTextPri)
	sCodeSpan  = lipgloss.NewStyle().Foreground(cTextAmber).Bold(true)

	sChipAmber = lipgloss.NewStyle().
			Foreground(cAccent).
			Background(lipgloss.Color("#2d1500")).
			Bold(true).
			Padding(0, 1)
	sChipGreen = lipgloss.NewStyle().
			Foreground(cGreen).
			Background(lipgloss.Color("#0f2a1a")).
			Bold(true).
			Padding(0, 1)
	sChipMuted = lipgloss.NewStyle().
			Foreground(cTextMuted).
			Background(cBgElevated).
			Padding(0, 1)

	sHeader    = lipgloss.NewStyle().Background(cBgSurface).Foreground(cTextPri)
	sStatusBar = lipgloss.NewStyle().Background(cBgSurface).Foreground(cTextMuted)
	sStatusKey = lipgloss.NewStyle().Background(cBgElevated).Foreground(cTextSec).Padding(0, 1)

	sInputFocus = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(cAccentDim).
			Background(cBgSurface).
			Padding(0, 1)

	sModal = lipgloss.NewStyle().
		Background(cBgSurface).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(cAccentDim).
		Padding(1, 2)

	sSelected   = lipgloss.NewStyle().Foreground(cAccent).Bold(true)
	sUnselected = lipgloss.NewStyle().Foreground(cTextSec)
	sHover      = lipgloss.NewStyle().Foreground(cTextPri).Background(cBgHover)

	sEmptyHex   = lipgloss.NewStyle().Foreground(cAccent).Bold(true)
	sEmptyTitle = lipgloss.NewStyle().Foreground(cTextPri).Bold(true)
	sEmptyBody  = lipgloss.NewStyle().Foreground(cTextMuted)
	sSpinner    = lipgloss.NewStyle().Foreground(cAccent)
)

// ─────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────

type Message struct {
	Role      string
	Content   string
	Timestamp time.Time
}

type Stats struct {
	Engine           string  `json:"engine"`
	Mode             string  `json:"mode"`
	VramUsedGb       float64 `json:"vram_used_gb"`
	VramTotalGb      float64 `json:"vram_total_gb"`
	ActiveSessions   int     `json:"active_sessions"`
	ModelLoaded      bool    `json:"model_loaded"`
	ModelIdleSeconds float64 `json:"model_idle_seconds"`
}

type screen int

const (
	screenChat screen = iota
	screenSettings
	screenHelp
)

// ─────────────────────────────────────────────────────────────────
// Tea message types
// ─────────────────────────────────────────────────────────────────

type msgStreamChunk string
type msgStreamDone  struct{ sessionID string }
type msgStreamErr   struct{ err error }
type msgStatsOK     Stats
type msgStatsFail   struct{}

// ─────────────────────────────────────────────────────────────────
// App model
// ─────────────────────────────────────────────────────────────────

type model struct {
	width, height int
	screen        screen

	// chat
	messages  []Message
	streamBuf strings.Builder
	sessionID string
	loading   bool

	// widgets
	spinner  spinner.Model
	viewport viewport.Model
	textarea textarea.Model
	vpReady  bool

	// settings
	engine         string
	chatMode       string
	settingsCursor int
	stats          Stats
	statsLoaded    bool

	// status bar
	statusMsg   string
	statusError bool

	// pointer to the running program (set before p.Run)
	prog *tea.Program
}

func newModel(prog *tea.Program) model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = sSpinner

	ta := textarea.New()
	ta.Placeholder = "Napisz wiadomość…"
	ta.Focus()
	ta.ShowLineNumbers = false
	ta.CharLimit = 0
	ta.SetWidth(80)
	ta.SetHeight(3)
	base := lipgloss.NewStyle().Background(cBgSurface).Foreground(cTextPri)
	ta.FocusedStyle.Base = base
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle().Background(cBgSurface)
	ta.FocusedStyle.Placeholder = lipgloss.NewStyle().Foreground(cTextMuted)
	ta.BlurredStyle = ta.FocusedStyle

	vp := viewport.New(80, 20)
	vp.Style = sBgFull

	return model{
		engine:   "transformers",
		chatMode: "general",
		spinner:  sp,
		viewport: vp,
		textarea: ta,
		prog:     prog,
	}
}

// ─────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────

func (m model) Init() tea.Cmd {
	return tea.Batch(
		textarea.Blink,
		m.spinner.Tick,
		cmdFetchStats(),
		tea.SetWindowTitle("HexAi v"+appVersion),
	)
}

// ─────────────────────────────────────────────────────────────────
// Update
// ─────────────────────────────────────────────────────────────────

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.reflow()
		m.vpReady = true
		m.viewport.SetContent(m.chatContent())
		return m, nil

	case spinner.TickMsg:
		var c tea.Cmd
		m.spinner, c = m.spinner.Update(msg)
		cmds = append(cmds, c)

	case msgStatsOK:
		s := Stats(msg)
		m.stats, m.statsLoaded = s, true
		m.engine, m.chatMode = s.Engine, s.Mode
		cmds = append(cmds, tea.Tick(5*time.Second, func(_ time.Time) tea.Msg {
			return doFetchStats()
		}))

	case msgStatsFail:
		cmds = append(cmds, tea.Tick(5*time.Second, func(_ time.Time) tea.Msg {
			return doFetchStats()
		}))

	case msgStreamChunk:
		m.streamBuf.WriteString(string(msg))
		c := m.streamBuf.String()
		if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
			m.messages[len(m.messages)-1].Content = c
		} else {
			m.messages = append(m.messages, Message{Role: "assistant", Content: c, Timestamp: time.Now()})
		}
		m.viewport.SetContent(m.chatContent())
		m.viewport.GotoBottom()

	case msgStreamDone:
		m.loading = false
		m.streamBuf.Reset()
		if msg.sessionID != "" {
			m.sessionID = msg.sessionID
		}
		m.viewport.SetContent(m.chatContent())
		m.viewport.GotoBottom()

	case msgStreamErr:
		m.loading = false
		m.streamBuf.Reset()
		m.messages = append(m.messages, Message{
			Role: "assistant", Content: "❌ Błąd: " + msg.err.Error(), Timestamp: time.Now(),
		})
		m.setStatus("Błąd połączenia z API", true)
		m.viewport.SetContent(m.chatContent())
		m.viewport.GotoBottom()

	case tea.KeyMsg:
		// ── Global shortcuts ──────────────────────────────────────
		switch msg.String() {
		case "ctrl+c", "ctrl+q":
			return m, tea.Quit

		case "ctrl+n":
			m.messages = nil
			m.sessionID = ""
			m.streamBuf.Reset()
			m.viewport.SetContent(m.chatContent())
			m.setStatus("Nowa rozmowa", false)
			return m, nil

		case "ctrl+s":
			if m.screen == screenSettings {
				m.screen = screenChat
			} else {
				m.screen = screenSettings
			}
			return m, nil

		case "esc":
			if m.screen != screenChat {
				m.screen = screenChat
				return m, nil
			}

		case "?":
			if m.screen == screenHelp {
				m.screen = screenChat
			} else if m.screen == screenChat {
				m.screen = screenHelp
			}
			return m, nil
		}

		// ── Per-screen routing ────────────────────────────────────
		switch m.screen {

		case screenHelp:
			m.screen = screenChat
			return m, nil

		case screenSettings:
			return m.handleSettingsKey(msg)

		case screenChat:
			// Enter → send message (do NOT pass to textarea)
			if msg.String() == "enter" && !m.loading {
				text := strings.TrimSpace(m.textarea.Value())
				if text != "" {
					m.textarea.Reset()
					m.messages = append(m.messages, Message{
						Role: "user", Content: text, Timestamp: time.Now(),
					})
					m.loading = true
					m.viewport.SetContent(m.chatContent())
					m.viewport.GotoBottom()
					cmds = append(cmds,
						cmdStream(m.sessionID, text, m.prog),
						m.spinner.Tick,
					)
					return m, tea.Batch(cmds...)
				}
				// empty enter – still don't pass to textarea
				return m, nil
			}
			// All other keys in chat: fall through to textarea update below
		}
	}

	// ── Textarea + viewport updates (chat mode only) ──────────────
	if m.screen == screenChat {
		var c tea.Cmd
		m.textarea, c = m.textarea.Update(msg)
		cmds = append(cmds, c)

		m.viewport, c = m.viewport.Update(msg)
		cmds = append(cmds, c)
	}

	return m, tea.Batch(cmds...)
}

func (m model) handleSettingsKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	const maxIdx = 3
	switch msg.String() {
	case "up", "k":
		if m.settingsCursor > 0 {
			m.settingsCursor--
		}
	case "down", "j":
		if m.settingsCursor < maxIdx {
			m.settingsCursor++
		}
	case "enter", " ":
		m.screen = screenChat
		switch m.settingsCursor {
		case 0:
			m.engine = "transformers"
			m.setStatus("Silnik: GPU (Transformers)", false)
			return m, cmdSetEngine("transformers")
		case 1:
			m.engine = "ollama"
			m.setStatus("Silnik: CPU (Ollama)", false)
			return m, cmdSetEngine("ollama")
		case 2:
			m.chatMode = "general"
			m.setStatus("Tryb: Ogólny", false)
			return m, cmdSetMode("general")
		case 3:
			m.chatMode = "programista"
			m.setStatus("Tryb: Programista", false)
			return m, cmdSetMode("programista")
		}
	}
	return m, nil
}

func (m *model) setStatus(s string, isErr bool) {
	m.statusMsg, m.statusError = s, isErr
}

func (m *model) reflow() {
	// header(1) + divider(1) + statusbar(1) + inputBox(~5) = ~8 reserved
	vpH := m.height - 9
	if vpH < 4 {
		vpH = 4
	}
	m.viewport.Width = m.width
	m.viewport.Height = vpH
	m.textarea.SetWidth(m.width - 6)
}

// ─────────────────────────────────────────────────────────────────
// View
// ─────────────────────────────────────────────────────────────────

func (m model) View() string {
	if m.width == 0 {
		return "Inicjalizacja…"
	}
	switch m.screen {
	case screenHelp:
		return m.viewOverlay(m.helpContent(), 46)
	case screenSettings:
		return m.viewOverlay(m.settingsContent(), 42)
	}
	return strings.Join([]string{
		m.renderHeader(),
		sDivider.Render(strings.Repeat("─", m.width)),
		m.viewport.View(),
		m.renderInput(),
		m.renderStatusBar(),
	}, "\n")
}

// ── Header ────────────────────────────────────────────────────────

func (m model) renderHeader() string {
	logo := sAccentB.Render("⬡ HexAi") + sMuted.Render("  v"+appVersion)

	var center string
	if m.sessionID != "" && len(m.sessionID) >= 8 {
		center = sMuted.Render("Sesja ") + sPrimary.Render(m.sessionID[:8]+"…")
	} else {
		center = sPrimary.Render("Nowa rozmowa")
	}

	engineLabel := "⚡ GPU"
	if m.engine == "ollama" {
		engineLabel = "🖥  CPU"
	}
	modeLabel := "Ogólny"
	if m.chatMode == "programista" {
		modeLabel = "Dev"
	}
	modelChip := sChipMuted.Render("○ offline")
	if m.statsLoaded && m.stats.ModelLoaded {
		modelChip = sChipGreen.Render("● model")
	}
	right := sChipAmber.Render(engineLabel) + "  " + sMuted.Render(modeLabel) + "  " + modelChip

	lw := lipgloss.Width(logo)
	cw := lipgloss.Width(center)
	rw := lipgloss.Width(right)
	gap := m.width - lw - cw - rw - 4
	if gap < 0 {
		gap = 0
	}
	lp := gap / 2
	rp := gap - lp

	line := logo + strings.Repeat(" ", lp) + center + strings.Repeat(" ", rp) + right
	return sHeader.Width(m.width).Padding(0, 2).Render(line)
}

// ── Input ─────────────────────────────────────────────────────────

func (m model) renderInput() string {
	w := m.width - 4

	var hint string
	if m.loading {
		hint = sAccent.Render(" " + m.spinner.View() + " Generowanie…")
	} else {
		hint = sMuted.Render(" Enter wyślij · Shift+Enter nowa linia · Ctrl+N nowa · ? pomoc")
	}

	inner := m.textarea.View() + "\n" + hint
	return "\n" + sInputFocus.Width(w).Render(inner)
}

// ── Status bar ────────────────────────────────────────────────────

func (m model) renderStatusBar() string {
	keys := [][2]string{
		{"Ctrl+N", "Nowa"},
		{"Ctrl+S", "Ustawienia"},
		{"?", "Pomoc"},
		{"Ctrl+C", "Wyjście"},
	}
	var parts []string
	for _, kv := range keys {
		parts = append(parts, sStatusKey.Render(kv[0])+sMuted.Render(" "+kv[1]))
	}
	left := strings.Join(parts, "  ")

	var right string
	if m.statusMsg != "" {
		if m.statusError {
			right = sError.Render(" " + m.statusMsg + " ")
		} else {
			right = sAccent.Render(" " + m.statusMsg + " ")
		}
	} else if m.statsLoaded && m.stats.VramTotalGb > 0 {
		pct := int(m.stats.VramUsedGb / m.stats.VramTotalGb * 100)
		right = sMuted.Render(fmt.Sprintf(" VRAM %d%% · %d sesji ", pct, m.stats.ActiveSessions))
	}

	lw := lipgloss.Width(left)
	rw := lipgloss.Width(right)
	gap := m.width - lw - rw
	if gap < 0 {
		gap = 0
	}
	return sStatusBar.Width(m.width).Render(left + strings.Repeat(" ", gap) + right)
}

// ── Chat content ──────────────────────────────────────────────────

func (m model) chatContent() string {
	if len(m.messages) == 0 {
		return strings.Join([]string{
			"",
			sEmptyHex.Render("        ⬡  HexAi"),
			"",
			sEmptyTitle.Render("        Jak mogę Ci dziś pomóc?"),
			"",
			sEmptyBody.Render("        Wpisz wiadomość i naciśnij Enter."),
			sEmptyBody.Render("        Użyj ? aby zobaczyć skróty klawiszowe."),
			"",
		}, "\n")
	}

	textW := m.width - 8
	if textW < 20 {
		textW = 20
	}

	var sb strings.Builder
	for i, msg := range m.messages {
		if i > 0 {
			sb.WriteString("\n")
		}
		ts := msg.Timestamp.Format("15:04")
		if msg.Role == "user" {
			sb.WriteString(sUserLabel.Render("  Ty") + sTimestamp.Render("  "+ts) + "\n")
			for _, line := range strings.Split(wrap.String(msg.Content, textW), "\n") {
				sb.WriteString(sUserText.Render("  "+line) + "\n")
			}
		} else {
			sb.WriteString(sAiLabel.Render("  HexAi") + sTimestamp.Render("  "+ts) + "\n")
			for _, line := range strings.Split(wrap.String(msg.Content, textW), "\n") {
				sb.WriteString(renderAiLine(line) + "\n")
			}
		}
		sb.WriteString("\n")
	}

	if m.loading && (len(m.messages) == 0 || m.messages[len(m.messages)-1].Role != "assistant") {
		sb.WriteString(sAiLabel.Render("  HexAi") + "\n")
		sb.WriteString(sAccent.Render("  "+m.spinner.View()+" myślę…") + "\n\n")
	}

	return sb.String()
}

func renderAiLine(line string) string {
	var out strings.Builder
	out.WriteString("  ")
	runes := []rune(line)
	var buf []rune
	inCode := false
	for _, r := range runes {
		if r == '`' {
			if inCode {
				out.WriteString(sCodeSpan.Render(string(buf)))
				buf = buf[:0]
				inCode = false
			} else {
				out.WriteString(sAiText.Render(string(buf)))
				buf = buf[:0]
				inCode = true
			}
		} else {
			buf = append(buf, r)
		}
	}
	if len(buf) > 0 {
		if inCode {
			out.WriteString(sCodeSpan.Render(string(buf)))
		} else {
			out.WriteString(sAiText.Render(string(buf)))
		}
	}
	return out.String()
}

// ── Overlay ───────────────────────────────────────────────────────

func (m model) viewOverlay(content string, modalW int) string {
	modal := sModal.Width(modalW).Render(content)
	mW := lipgloss.Width(modal)
	mH := strings.Count(modal, "\n") + 1
	px := (m.width - mW) / 2
	py := (m.height - mH) / 2
	if px < 0 {
		px = 0
	}
	if py < 0 {
		py = 0
	}
	blank := sBgFull.Width(m.width).Render("")
	var full strings.Builder
	for i := 0; i < py; i++ {
		full.WriteString(blank + "\n")
	}
	for _, line := range strings.Split(modal, "\n") {
		full.WriteString(strings.Repeat(" ", px) + line + "\n")
	}
	for i := mH + py; i < m.height; i++ {
		full.WriteString(blank + "\n")
	}
	return full.String()
}

func (m model) settingsContent() string {
	type item struct {
		group, value, label string
		idx                 int
	}
	items := []item{
		{"SILNIK", "transformers", "⚡ Transformers (GPU)", 0},
		{"SILNIK", "ollama", "🖥  Ollama (CPU)", 1},
		{"TRYB", "general", "💬 Ogólny", 2},
		{"TRYB", "programista", "💻 Programista (Dev)", 3},
	}

	var sb strings.Builder
	sb.WriteString(sAccentB.Render("  ⚙  Ustawienia") + "\n")
	sb.WriteString(sDivider.Render(strings.Repeat("─", 34)) + "\n\n")

	prev := ""
	for _, it := range items {
		if it.group != prev {
			sb.WriteString(sMuted.Render("  "+it.group) + "\n")
			prev = it.group
		}
		isCurrent := (it.group == "SILNIK" && it.value == m.engine) ||
			(it.group == "TRYB" && it.value == m.chatMode)
		isCursor := m.settingsCursor == it.idx

		var s lipgloss.Style
		prefix := "  ○ "
		if isCurrent {
			prefix = "  ● "
			s = sSelected
		} else if isCursor {
			s = sHover
		} else {
			s = sUnselected
		}
		sb.WriteString(s.Render(prefix+it.label) + "\n")
	}

	if m.statsLoaded && m.stats.VramTotalGb > 0 {
		pct := int(m.stats.VramUsedGb / m.stats.VramTotalGb * 100)
		bc := cGreen
		if pct > 80 {
			bc = cRed
		} else if pct > 60 {
			bc = cAccent
		}
		filled := pct * 20 / 100
		bar := lipgloss.NewStyle().Foreground(bc).
			Render(strings.Repeat("█", filled) + strings.Repeat("░", 20-filled))
		sb.WriteString("\n" + sMuted.Render("  VRAM") + "\n")
		sb.WriteString("  " + bar + sMuted.Render(
			fmt.Sprintf(" %d%% (%.1f/%.1fGB)", pct, m.stats.VramUsedGb, m.stats.VramTotalGb),
		) + "\n")
	}

	sb.WriteString("\n" + sDivider.Render(strings.Repeat("─", 34)) + "\n")
	sb.WriteString(sMuted.Render("  ↑↓/jk Nawiguj · Enter Wybierz · Esc Zamknij"))
	return sb.String()
}

func (m model) helpContent() string {
	shortcuts := [][2]string{
		{"Enter", "Wyślij wiadomość"},
		{"Shift+Enter", "Nowa linia"},
		{"Ctrl+N", "Nowa rozmowa"},
		{"Ctrl+S", "Ustawienia"},
		{"↑↓ / PgUp/PgDn", "Przewijaj historię"},
		{"?", "Ta pomoc"},
		{"Ctrl+C / Ctrl+Q", "Wyjście"},
	}
	var sb strings.Builder
	sb.WriteString(sAccentB.Render("  ⬡  HexAi – Pomoc") + "\n")
	sb.WriteString(sDivider.Render(strings.Repeat("─", 38)) + "\n\n")
	for _, kv := range shortcuts {
		k := sAccent.Bold(true).Width(17).Render(kv[0])
		sb.WriteString("  " + k + "  " + sAiText.Render(kv[1]) + "\n")
	}
	sb.WriteString("\n" + sDivider.Render(strings.Repeat("─", 38)) + "\n")
	sb.WriteString(sMuted.Render("  API: http://localhost:8000") + "\n")
	sb.WriteString(sMuted.Render("  HexAi dla HackerOS · GPL-3.0") + "\n\n")
	sb.WriteString(sMuted.Render("  Naciśnij dowolny klawisz aby zamknąć"))
	return sb.String()
}

// ─────────────────────────────────────────────────────────────────
// Network
// ─────────────────────────────────────────────────────────────────

func cmdFetchStats() tea.Cmd {
	return func() tea.Msg { return doFetchStats() }
}

func doFetchStats() tea.Msg {
	resp, err := http.Get(apiBase + "/stats")
	if err != nil {
		return msgStatsFail{}
	}
	defer resp.Body.Close()
	var s Stats
	if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
		return msgStatsFail{}
	}
	return msgStatsOK(s)
}

// cmdStream spawns a goroutine that reads the SSE-style stream and
// pushes chunks to the program via p.Send().
func cmdStream(sessionID, message string, p *tea.Program) tea.Cmd {
	return func() tea.Msg {
		go func() {
			payload := map[string]interface{}{"message": message, "stream": true}
			if sessionID != "" {
				payload["session_id"] = sessionID
			}
			data, _ := json.Marshal(payload)

			req, err := http.NewRequest("POST", apiBase+"/chat", bytes.NewReader(data))
			if err != nil {
				p.Send(msgStreamErr{err})
				return
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				p.Send(msgStreamErr{err})
				return
			}
			defer resp.Body.Close()

			r := bufio.NewReaderSize(resp.Body, 512)
			buf := make([]byte, 512)
			for {
				n, err := r.Read(buf)
				if n > 0 {
					chunk := make([]byte, n)
					copy(chunk, buf[:n])
					p.Send(msgStreamChunk(chunk))
				}
				if err == io.EOF {
					break
				}
				if err != nil {
					p.Send(msgStreamErr{err})
					return
				}
			}
			p.Send(msgStreamDone{})
		}()
		return nil // Cmd returns immediately; goroutine does the work
	}
}

func cmdSetEngine(e string) tea.Cmd {
	return func() tea.Msg {
		data, _ := json.Marshal(map[string]string{"engine": e})
		http.Post(apiBase+"/engine", "application/json", bytes.NewReader(data)) //nolint
		return nil
	}
}

func cmdSetMode(mode string) tea.Cmd {
	return func() tea.Msg {
		data, _ := json.Marshal(map[string]string{"mode": mode})
		http.Post(apiBase+"/mode", "application/json", bytes.NewReader(data)) //nolint
		return nil
	}
}

// ─────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────

func main() {
	// We need the program reference inside the model before it starts,
	// so we create a placeholder, build the model, then create the real
	// program with that model (the pointer is already set correctly).
	var p *tea.Program
	m := newModel(nil) // prog set to nil first

	p = tea.NewProgram(
		m,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	// Patch the prog pointer: Bubble Tea copies the model value on start,
	// but m.prog is a *tea.Program (pointer). We need to make sure the
	// model that gets copied has the right pointer. The cleanest way is
	// to recreate after we have p:
	m.prog = p
	p = tea.NewProgram(
		m,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Błąd: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Do widzenia! 👋")
}
