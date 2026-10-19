import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:speech_to_text/speech_to_text.dart' as stt;

import '../api.dart';
import '../services/app_language.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';

/// The authenticated StayBuddy AI assistant. The backend supplies the RAG
/// answer and only role-authorized account information for the current user.
class ChatbotScreen extends StatefulWidget {
  const ChatbotScreen({super.key});

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  final Api _api = Api();
  final _composer = TextEditingController();
  final _scroll = ScrollController();
  final stt.SpeechToText _speech = stt.SpeechToText();
  final FlutterTts _tts = FlutterTts();
  List<Map<String, dynamic>> _sessions = const [];
  List<Map<String, dynamic>> _messages = const [];
  String? _sessionId;
  String? _role;
  String? _error;
  bool _loading = true;
  bool _sending = false;
  bool _listening = false;
  bool _speakReplies = false;
  String _provider = 'ollama';
  http.Client? _activeClient;
  int _sequence = 0;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  @override
  void dispose() {
    _activeClient?.close();
    _speech.stop();
    _tts.stop();
    _composer.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    final role = await AuthStore.getRole();
    if (!mounted) return;
    if (!const {'student', 'parent', 'warden', 'owner'}.contains(role)) {
      setState(() {
        _role = role;
        _loading = false;
        _error = 'Sign in as a student, parent, warden, or owner to use your personal assistant.';
      });
      return;
    }
    setState(() => _role = role);
    await _loadSessions(openFirst: true);
  }

  Future<void> _loadSessions({bool openFirst = false}) async {
    try {
      final records = await _api.getChatSessions();
      final sessions = records.map((item) => Map<String, dynamic>.from(item as Map)).toList();
      if (!mounted) return;
      setState(() {
        _sessions = sessions;
        _loading = false;
        _error = null;
      });
      if (openFirst && sessions.isNotEmpty) await _openSession(sessions.first['id'].toString());
    } catch (error) {
      if (mounted) setState(() { _loading = false; _error = _messageFor(error); });
    }
  }

  Future<void> _openSession(String id) async {
    if (_sending) return;
    try {
      setState(() => _loading = true);
      final records = await _api.getChatMessages(id);
      if (!mounted) return;
      setState(() {
        _sessionId = id;
        _messages = records.map((item) => Map<String, dynamic>.from(item as Map)).toList();
        _loading = false;
      });
      _scrollToBottom();
    } catch (error) {
      if (mounted) setState(() { _loading = false; _error = _messageFor(error); });
    }
  }

  Future<void> _newChat() async {
    if (_sending) return;
    try {
      setState(() => _loading = true);
      final result = await _api.createChatSession();
      final session = Map<String, dynamic>.from(result['session'] as Map);
      if (!mounted) return;
      setState(() {
        _sessions = [session, ..._sessions];
        _sessionId = session['id'].toString();
        _messages = const [];
        _loading = false;
      });
    } catch (error) {
      if (mounted) setState(() { _loading = false; _error = _messageFor(error); });
    }
  }

  Future<void> _rename(Map<String, dynamic> session) async {
    final controller = TextEditingController(text: session['title']?.toString() ?? '');
    final title = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Rename chat'),
        content: TextField(controller: controller, autofocus: true, maxLength: 160),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text.trim()), child: const Text('Save')),
        ],
      ),
    );
    controller.dispose();
    if (title == null || title.isEmpty) return;
    try {
      final result = await _api.renameChatSession(session['id'].toString(), title);
      final updated = Map<String, dynamic>.from(result['session'] as Map);
      if (!mounted) return;
      setState(() => _sessions = _sessions.map((item) => item['id'].toString() == updated['id'].toString() ? updated : item).toList());
    } catch (error) {
      _showError(_messageFor(error));
    }
  }

  Future<void> _delete(Map<String, dynamic> session) async {
    if (!await _confirm('Delete this chat and all messages? This cannot be undone.')) return;
    try {
      final id = session['id'].toString();
      await _api.deleteChatSession(id);
      if (!mounted) return;
      setState(() {
        _sessions = _sessions.where((item) => item['id'].toString() != id).toList();
        if (_sessionId == id) {
          _sessionId = null;
          _messages = const [];
        }
      });
    } catch (error) {
      _showError(_messageFor(error));
    }
  }

  Future<void> _send() async {
    final text = _composer.text.trim();
    if (text.isEmpty || _sending) return;
    if (_sessionId == null) await _newChat();
    final sessionId = _sessionId;
    if (sessionId == null || !mounted) return;

    final requestId = 'flutter-${DateTime.now().microsecondsSinceEpoch}-${Random().nextInt(1 << 20)}';
    final pendingId = 'pending-$requestId';
    final sequence = ++_sequence;
    final client = http.Client();
    setState(() {
      _composer.clear();
      _messages = [..._messages,
        {'id': requestId, 'role': 'user', 'content': text},
        {'id': pendingId, 'role': 'assistant', 'content': 'Thinking...', 'pending': true},
      ];
      _sending = true;
      _error = null;
      _activeClient = client;
    });
    _scrollToBottom();

    try {
      final result = await _api.sendChatMessage(
        sessionId: sessionId,
        message: text,
        requestId: requestId,
        provider: _provider,
        language: AppLanguage.chatLanguageCode,
        client: client,
      );
      if (!mounted || sequence != _sequence) return;
      final assistant = Map<String, dynamic>.from(result['assistant_message'] as Map? ?? const {});
      final existingMetadata = assistant['metadata'] is Map
          ? Map<String, dynamic>.from(assistant['metadata'] as Map)
          : <String, dynamic>{};
      assistant['metadata'] = {...existingMetadata, ...result};
      setState(() => _messages = [..._messages.where((item) => item['id'] != pendingId), assistant]);
      if (_speakReplies) await _speak(assistant['content']?.toString() ?? '');
      await _loadSessions();
    } catch (error) {
      if (!mounted || sequence != _sequence) return;
      final stopped = _activeClient == null;
      setState(() => _messages = _messages.map((item) => item['id'] == pendingId
          ? {...item, 'content': stopped ? 'Response stopped.' : 'Could not get an answer: ${_messageFor(error)}', 'pending': false, 'error': !stopped}
          : item).toList());
    } finally {
      if (identical(_activeClient, client)) {
        client.close();
        if (mounted) setState(() { _activeClient = null; _sending = false; });
      }
    }
  }

  void _stop() {
    if (!_sending) return;
    ++_sequence;
    _activeClient?.close();
    setState(() {
      _activeClient = null;
      _sending = false;
      _messages = _messages.map((item) => item['pending'] == true
          ? {...item, 'content': 'Response stopped.', 'pending': false}
          : item).toList();
    });
  }

  Future<void> _toggleVoiceInput() async {
    if (_sending) return;
    if (_listening) {
      await _speech.stop();
      if (mounted) setState(() => _listening = false);
      return;
    }
    final available = await _speech.initialize(
      onStatus: (status) {
        if (mounted && (status == 'done' || status == 'notListening')) {
          setState(() => _listening = false);
        }
      },
      onError: (error) {
        if (mounted) {
          setState(() => _listening = false);
          _showError('Voice input error: ${error.errorMsg}');
        }
      },
    );
    if (!mounted) return;
    if (!available) {
      _showError('Voice input is unavailable. Allow microphone access and use a supported device or browser.');
      return;
    }
    setState(() => _listening = true);
    await _speech.listen(
      localeId: 'en_US',
      partialResults: true,
      onResult: (result) {
        if (!mounted) return;
        _composer.value = _composer.value.copyWith(
          text: result.recognizedWords,
          selection: TextSelection.collapsed(offset: result.recognizedWords.length),
        );
        if (result.finalResult && result.recognizedWords.trim().isNotEmpty) {
          _send();
        }
      },
    );
  }

  Future<void> _speak(String text) async {
    if (text.trim().isEmpty) return;
    await _tts.stop();
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.45);
    await _tts.speak(text);
  }

  Future<void> _performAction(Map<String, dynamic> action) async {
    final actionBody = action['body'] is Map ? Map<String, dynamic>.from(action['body'] as Map) : <String, dynamic>{};
    for (final requiredField in List<String>.from(action['required_fields'] as List? ?? const [])) {
      final answer = await _prompt('Enter ${requiredField.replaceAll('_', ' ')}');
      if (answer == null || answer.trim().isEmpty) return;
      actionBody[requiredField] = answer.trim();
    }
    if (!await _confirm('Confirm: ${action['label'] ?? 'this action'}?')) return;
    try {
      await _api.executeAssistantAction({...action, 'body': actionBody});
      _showSuccess('Action completed.');
    } catch (error) {
      _showError(_messageFor(error));
    }
  }

  Future<void> _feedback(int id, bool helpful) async {
    try {
      await _api.submitChatFeedback(id, helpful: helpful);
      _showSuccess('Thanks for your feedback.');
    } catch (error) {
      _showError(_messageFor(error));
    }
  }

  Future<bool> _confirm(String text) async => await showDialog<bool>(
    context: context,
    builder: (context) => AlertDialog(title: const Text('Confirm action'), content: Text(text), actions: [
      TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
      FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Confirm')),
    ]),
  ) ?? false;

  Future<String?> _prompt(String label) async {
    final controller = TextEditingController();
    final value = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(title: Text(label), content: TextField(controller: controller, autofocus: true, maxLines: 3), actions: [
        TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('Continue')),
      ]),
    );
    controller.dispose();
    return value;
  }

  void _scrollToBottom() => WidgetsBinding.instance.addPostFrameCallback((_) {
    if (_scroll.hasClients) _scroll.animateTo(_scroll.position.maxScrollExtent, duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
  });

  String _messageFor(Object error) => error.toString().replaceFirst('Exception: ', '');
  void _showError(String text) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text), backgroundColor: Colors.red.shade700));
  void _showSuccess(String text) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text), backgroundColor: AppTheme.teal));

  @override
  Widget build(BuildContext context) => ResponsiveShell(
    child: Scaffold(
      backgroundColor: AppTheme.bg,
      appBar: AppBar(
        // This screen is opened from each actor's dashboard.  An explicit
        // back button takes the user to that same dashboard; the chat drawer
        // remains available through the menu button on the right.
        automaticallyImplyLeading: false,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          tooltip: 'Back to dashboard',
          onPressed: () => Navigator.of(context).maybePop(),
        ),
        title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('StayBuddy AI Assistant'),
          Text(_role == null ? 'Loading account...' : '${_role![0].toUpperCase()}${_role!.substring(1)} assistant', style: const TextStyle(fontSize: 11)),
        ]),
        actions: [
          if (_role != null)
            Builder(builder: (buttonContext) => IconButton(
              onPressed: () => Scaffold.of(buttonContext).openDrawer(),
              tooltip: 'Saved chats',
              icon: const Icon(Icons.menu),
            )),
          IconButton(onPressed: _sending ? _stop : _newChat, tooltip: _sending ? 'Stop response' : 'New chat', icon: Icon(_sending ? Icons.stop_circle_outlined : Icons.add_comment_outlined)),
          IconButton(
            onPressed: () async {
              final next = !_speakReplies;
              if (!next) await _tts.stop();
              if (mounted) setState(() => _speakReplies = next);
            },
            tooltip: _speakReplies ? 'Turn off spoken replies' : 'Turn on spoken replies',
            icon: Icon(_speakReplies ? Icons.volume_up_rounded : Icons.volume_off_rounded),
          ),
        ],
      ),
      drawer: _role == null ? null : _drawer(),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _role == null || (_error != null && _sessions.isEmpty && _messages.isEmpty)
              ? _errorView()
              : Column(children: [
                  if (_error != null) _errorBanner(),
                  _providerBar(),
                  Expanded(child: _messages.isEmpty ? _emptyChat() : _chatList()),
                  _composerBar(),
                ]),
    ),
  );

  Widget _drawer() => Drawer(child: SafeArea(child: Column(children: [
    Padding(padding: const EdgeInsets.all(16), child: Row(children: [
      const Expanded(child: Text('Your chats', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700))),
      IconButton(onPressed: _newChat, icon: const Icon(Icons.add)),
    ])),
    Expanded(child: _sessions.isEmpty
        ? const Center(child: Text('No saved chats yet.'))
        : ListView.builder(itemCount: _sessions.length, itemBuilder: (context, index) {
            final session = _sessions[index];
            return ListTile(
              selected: session['id'].toString() == _sessionId,
              leading: const Icon(Icons.chat_bubble_outline),
              title: Text(session['title']?.toString() ?? 'Untitled chat', maxLines: 1, overflow: TextOverflow.ellipsis),
              onTap: () { Navigator.pop(context); _openSession(session['id'].toString()); },
              trailing: PopupMenuButton<String>(onSelected: (value) => value == 'rename' ? _rename(session) : _delete(session), itemBuilder: (_) => const [
                PopupMenuItem(value: 'rename', child: Text('Rename')),
                PopupMenuItem(value: 'delete', child: Text('Delete')),
              ]),
            );
          })),
  ])));

  Widget _errorView() => Center(child: Padding(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, children: [
    const Icon(Icons.lock_outline, size: 42, color: AppTheme.teal), const SizedBox(height: 12),
    Text(_error ?? 'Unable to open assistant.', textAlign: TextAlign.center), const SizedBox(height: 14),
    FilledButton(onPressed: _initialize, child: const Text('Try again')),
  ])));

  Widget _errorBanner() => Container(width: double.infinity, color: Colors.red.shade50, padding: const EdgeInsets.all(10), child: Text(_error!, style: TextStyle(color: Colors.red.shade800)));

  Widget _providerBar() => Padding(padding: const EdgeInsets.fromLTRB(16, 10, 16, 4), child: Row(children: [
    const Icon(Icons.privacy_tip_outlined, size: 17, color: AppTheme.teal), const SizedBox(width: 7),
    const Expanded(child: Text('Answers use only records you are authorized to access.', style: TextStyle(fontSize: 12))),
    DropdownButton<String>(value: _provider, underline: const SizedBox(), items: const [
      DropdownMenuItem(value: 'ollama', child: Text('Ollama')),
      DropdownMenuItem(value: 'groq', child: Text('Groq')),
    ], onChanged: _sending ? null : (value) => setState(() => _provider = value ?? 'ollama')),
  ]));

  Widget _emptyChat() {
    final suggestions = switch (_role) {
      'warden' => ['How many students are present today?', 'Show pending attendance appeals', 'Show the booking queue'],
      'owner' => ['How many residents are in my hostels?', 'Show my fee summary', 'Enable attendance for my hostel'],
      'parent' => ["When is my child's next fee due?", "Show my child's attendance", 'Notify me when my child returns'],
      _ => ['How do I book a hostel?', 'Show my due payments', 'What is my attendance status?'],
    };
    return Center(child: SingleChildScrollView(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, children: [
      Container(width: 62, height: 62, decoration: BoxDecoration(color: AppTheme.teal.withOpacity(.12), shape: BoxShape.circle), child: const Icon(Icons.auto_awesome, color: AppTheme.teal, size: 30)),
      const SizedBox(height: 16), const Text('How can I help?', style: TextStyle(fontSize: 21, fontWeight: FontWeight.w700)),
      const SizedBox(height: 8), const Text('Ask about StayBuddy or your own authorized records.', textAlign: TextAlign.center),
      const SizedBox(height: 22),
      ...suggestions.map((item) => Padding(padding: const EdgeInsets.only(bottom: 9), child: OutlinedButton(onPressed: () { _composer.text = item; _send(); }, child: Text(item)))),
    ])));
  }

  Widget _chatList() => ListView.builder(controller: _scroll, padding: const EdgeInsets.fromLTRB(14, 8, 14, 14), itemCount: _messages.length, itemBuilder: (context, index) => _bubble(_messages[index]));

  Widget _bubble(Map<String, dynamic> message) {
    final assistant = message['role'] == 'assistant';
    final metadata = message['metadata'] is Map ? Map<String, dynamic>.from(message['metadata'] as Map) : <String, dynamic>{};
    final actions = (metadata['actions'] as List? ?? const []).whereType<Map>().map((item) => Map<String, dynamic>.from(item)).where((item) => item['type'] == 'api').toList();
    final id = int.tryParse(message['id']?.toString() ?? '');
    return Align(alignment: assistant ? Alignment.centerLeft : Alignment.centerRight, child: Container(
      constraints: const BoxConstraints(maxWidth: 650), margin: const EdgeInsets.only(bottom: 12), padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(color: assistant ? Colors.white : AppTheme.teal, borderRadius: BorderRadius.circular(16), border: assistant ? Border.all(color: AppTheme.border) : null),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(message['content']?.toString() ?? '', style: TextStyle(height: 1.45, color: assistant ? AppTheme.textPrimary : Colors.white)),
        if (assistant && message['pending'] == true) const Padding(padding: EdgeInsets.only(top: 8), child: SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))),
        if (assistant && actions.isNotEmpty) ...[
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8, children: actions.map((action) => OutlinedButton.icon(onPressed: () => _performAction(action), icon: const Icon(Icons.verified_user_outlined, size: 16), label: Text(action['label']?.toString() ?? 'Confirm action'))).toList()),
        ],
        if (assistant && id != null && message['pending'] != true) Row(mainAxisSize: MainAxisSize.min, children: [
          IconButton(tooltip: 'Helpful', onPressed: () => _feedback(id, true), icon: const Icon(Icons.thumb_up_outlined, size: 17)),
          IconButton(tooltip: 'Not helpful', onPressed: () => _feedback(id, false), icon: const Icon(Icons.thumb_down_outlined, size: 17)),
        ]),
      ]),
    ));
  }

  Widget _composerBar() => SafeArea(top: false, child: Container(padding: const EdgeInsets.fromLTRB(12, 8, 12, 12), decoration: const BoxDecoration(color: Colors.white, border: Border(top: BorderSide(color: AppTheme.border))), child: Row(children: [
    Expanded(child: TextField(controller: _composer, minLines: 1, maxLines: 5, enabled: !_sending, textInputAction: TextInputAction.send, onSubmitted: (_) => _send(), decoration: const InputDecoration(hintText: 'Ask your StayBuddy assistant...', border: InputBorder.none, filled: false))),
    IconButton(onPressed: _sending ? null : _toggleVoiceInput, icon: Icon(_listening ? Icons.stop_circle_outlined : Icons.mic_none_rounded), tooltip: _listening ? 'Stop listening' : 'Speak your question'),
    const SizedBox(width: 8), IconButton.filled(onPressed: _sending ? _stop : _send, icon: Icon(_sending ? Icons.stop : Icons.send), tooltip: _sending ? 'Stop response' : 'Send'),
  ])));
}
