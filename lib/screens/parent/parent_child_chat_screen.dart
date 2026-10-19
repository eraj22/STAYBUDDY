import 'dart:async';

import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

import '../../api.dart';

/// Authenticated parent-to-linked-student chat. The periodic refresh is a
/// fallback for clients that cannot keep the backend SSE stream open.
class ParentChildChatScreen extends StatefulWidget {
  const ParentChildChatScreen({super.key});

  @override
  State<ParentChildChatScreen> createState() => _ParentChildChatScreenState();
}

class _ParentChildChatScreenState extends State<ParentChildChatScreen> {
  final _messageController = TextEditingController();
  final _scrollController = ScrollController();
  final stt.SpeechToText _speech = stt.SpeechToText();
  Timer? _refreshTimer;
  int? _conversationId;
  String _childName = 'Child';
  List<dynamic> _messages = const [];
  String? _error;
  bool _loading = true;
  bool _sending = false;
  bool _listening = false;

  @override
  void initState() {
    super.initState();
    _loadConversation();
    _refreshTimer = Timer.periodic(const Duration(seconds: 8), (_) => _refreshMessages());
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _speech.stop();
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _loadConversation() async {
    setState(() { _loading = true; _error = null; });
    try {
      final children = await Api().getParentChildren();
      if (children.isEmpty) throw Exception('No linked student is available for chat.');
      final child = Map<String, dynamic>.from(children.first as Map);
      final studentId = int.tryParse(child['id']?.toString() ?? '');
      if (studentId == null) throw Exception('The linked student record is invalid.');
      final conversation = await Api().openParentStudentConversation(studentId);
      _conversationId = int.tryParse(conversation['id']?.toString() ?? '');
      _childName = child['name']?.toString() ?? 'Child';
      if (_conversationId == null) throw Exception('The conversation could not be opened.');
      await _refreshMessages(showLoading: false);
    } catch (error) {
      if (mounted) setState(() => _error = error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _refreshMessages({bool showLoading = false}) async {
    final id = _conversationId;
    if (id == null) return;
    try {
      final messages = await Api().getConversationMessages(id);
      if (mounted) setState(() { _messages = messages; _error = null; });
    } catch (error) {
      if (mounted && showLoading) setState(() => _error = error.toString().replaceFirst('Exception: ', ''));
    }
  }

  Future<void> _send() async {
    final body = _messageController.text.trim();
    final id = _conversationId;
    if (body.isEmpty || id == null || _sending) return;
    setState(() => _sending = true);
    try {
      await Api().sendConversationMessage(conversationId: id, body: body);
      _messageController.clear();
      await _refreshMessages();
      if (_scrollController.hasClients) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
        if (_scrollController.hasClients) _scrollController.animateTo(_scrollController.position.maxScrollExtent, duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
      }
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(error.toString().replaceFirst('Exception: ', ''))));
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _toggleVoiceInput() async {
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
      onError: (_) {
        if (mounted) setState(() => _listening = false);
      },
    );
    if (!available) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Voice input is unavailable. Allow microphone permission and try again.')));
      return;
    }
    setState(() => _listening = true);
    await _speech.listen(
      onResult: (result) {
        if (!mounted) return;
        setState(() => _messageController.text = result.recognizedWords);
      },
      listenFor: const Duration(seconds: 45),
      pauseFor: const Duration(seconds: 4),
    );
  }

  String _time(dynamic value) {
    final date = DateTime.tryParse(value?.toString() ?? '')?.toLocal();
    if (date == null) return '';
    final hour = date.hour % 12 == 0 ? 12 : date.hour % 12;
    return '$hour:${date.minute.toString().padLeft(2, '0')} ${date.hour >= 12 ? 'PM' : 'AM'}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Chat with $_childName')),
      body: Column(children: [
        if (_loading) const Expanded(child: Center(child: CircularProgressIndicator()))
        else if (_error != null) Expanded(child: Center(child: Padding(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, children: [Text(_error!, textAlign: TextAlign.center), const SizedBox(height: 12), FilledButton(onPressed: _loadConversation, child: const Text('Retry'))]))))
        else Expanded(child: _messages.isEmpty ? const Center(child: Text('No messages yet. Start the conversation.')) : ListView.builder(
          controller: _scrollController, padding: const EdgeInsets.all(16), itemCount: _messages.length,
          itemBuilder: (_, index) {
            final message = Map<String, dynamic>.from(_messages[index] as Map);
            final mine = message['is_mine'] == true;
            return Align(alignment: mine ? Alignment.centerRight : Alignment.centerLeft, child: Container(
              margin: const EdgeInsets.only(bottom: 10), padding: const EdgeInsets.all(12), constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * .76),
              decoration: BoxDecoration(color: mine ? Theme.of(context).colorScheme.primaryContainer : Theme.of(context).colorScheme.surfaceContainerHighest, borderRadius: BorderRadius.circular(14)),
              child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [Align(alignment: Alignment.centerLeft, child: Text(message['body']?.toString() ?? '')), const SizedBox(height: 3), Text(_time(message['created_at']), style: Theme.of(context).textTheme.labelSmall)]),
            ));
          },
        )),
        SafeArea(top: false, child: Padding(padding: const EdgeInsets.all(10), child: Row(children: [
          Expanded(child: TextField(controller: _messageController, minLines: 1, maxLines: 4, onSubmitted: (_) => _send(), decoration: const InputDecoration(hintText: 'Type a message', border: OutlineInputBorder()))),
          IconButton(onPressed: _sending ? null : _toggleVoiceInput, tooltip: _listening ? 'Stop listening' : 'Speak your message', icon: Icon(_listening ? Icons.stop_circle_outlined : Icons.mic_none_rounded)),
          const SizedBox(width: 8), IconButton(onPressed: _sending ? null : _send, icon: _sending ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.send)),
        ]))),
      ]),
    );
  }
}
