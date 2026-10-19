import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

import '../../api.dart';
import '../chatbot_screen.dart';
import 'parent_admin_chat_screen.dart';
import 'parent_announcements_screen.dart';
import 'parent_attendance_screen.dart';
import 'parent_emergency_screen.dart';
import 'parent_fees_screen.dart';
import 'parent_location_screen.dart';
import 'parent_transport_screen.dart';

/// English-only speech controls for the parent portal. Speech is handled by
/// the device/browser speech service and is not stored by StayBuddy.
class ParentVoiceCommandScreen extends StatefulWidget {
  const ParentVoiceCommandScreen({super.key});

  @override
  State<ParentVoiceCommandScreen> createState() => _ParentVoiceCommandScreenState();
}

class _ParentVoiceCommandScreenState extends State<ParentVoiceCommandScreen> {
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _listening = false;
  bool _available = false;
  String _recognizedText = 'Tap the microphone and speak an English command.';

  @override
  void dispose() {
    _speech.stop();
    super.dispose();
  }

  Future<void> _toggleListening() async {
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
        if (mounted) setState(() {
          _listening = false;
          _recognizedText = 'Voice input error: ${error.errorMsg}';
        });
      },
    );
    if (!mounted) return;
    if (!available) {
      setState(() => _recognizedText =
          'Speech recognition is unavailable. Allow microphone access and use a supported browser or device.');
      return;
    }
    setState(() {
      _available = true;
      _listening = true;
      _recognizedText = 'Listening…';
    });
    await _speech.listen(
      localeId: 'en_US',
      partialResults: true,
      onResult: (result) {
        if (!mounted) return;
        setState(() => _recognizedText = result.recognizedWords.isEmpty ? 'Listening…' : result.recognizedWords);
        if (result.finalResult && result.recognizedWords.trim().isNotEmpty) {
          _runCommand(result.recognizedWords);
        }
      },
    );
  }

  Future<void> _runCommand(String command) async {
    await _speech.stop();
    if (mounted) setState(() => _listening = false);
    final text = command.toLowerCase();
    Widget? target;
    String? message;
    try {
      final result = await Api().runParentVoiceCommand(command);
      final intent = result['intent']?.toString();
      final response = result['response_text']?.toString();
      if (intent == 'child_location') {
        target = const ParentLocationScreen();
      } else if (intent == 'child_fees') {
        target = const ParentFeesScreen();
      } else if (intent == 'child_attendance') {
        target = const ParentAttendanceScreen();
      } else if (intent == 'child_announcements') {
        target = const ParentAnnouncementsScreen();
      } else if (intent == 'child_emergencies') {
        target = const ParentEmergencyScreen();
      }
      message = response;
    } catch (_) {
      // Screen navigation remains available when the parent record query is
      // temporarily unavailable, for example during a network interruption.
    }
    if (target == null && (text.contains('location') || text.contains('where'))) {
      target = const ParentLocationScreen(); message = 'Opening live location.';
    } else if (target == null && (text.contains('fee') || text.contains('payment') || text.contains('invoice'))) {
      target = const ParentFeesScreen(); message = 'Opening fee details.';
    } else if (target == null && (text.contains('attendance') || text.contains('present') || text.contains('in and out'))) {
      target = const ParentAttendanceScreen(); message = 'Opening attendance.';
    } else if (target == null && (text.contains('transport') || text.contains('bus') || text.contains('van'))) {
      target = const ParentTransportScreen(); message = 'Opening transport details.';
    } else if (target == null && (text.contains('announcement') || text.contains('notice'))) {
      target = const ParentAnnouncementsScreen(); message = 'Opening announcements.';
    } else if (target == null && (text.contains('emergency') || text.contains('sos') || text.contains('help'))) {
      target = const ParentEmergencyScreen(); message = 'Opening emergency updates.';
    } else if (target == null && (text.contains('admin') || text.contains('warden') || text.contains('staff'))) {
      target = const ParentAdminChatScreen(); message = 'Opening hostel staff chat.';
    } else if (target == null && (text.contains('assistant') || text.contains('chatbot') || text.contains('chat bot'))) {
      target = const ChatbotScreen(); message = 'Opening the AI assistant.';
    }
    if (!mounted) return;
    if (target == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('I did not recognise that command. Try location, fees, attendance, transport, announcements, emergency, staff, or AI assistant.')));
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message ?? 'Opening service.')));
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => target!));
  }

  @override
  Widget build(BuildContext context) {
    const examples = <String>[
      'Show my child location', 'Open fee details', 'Show attendance',
      'Show transport details', 'Show announcements', 'Open AI assistant',
    ];
    return Scaffold(
      appBar: AppBar(title: const Text('Voice Commands')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Text('Speak an English command to open a parent service.', style: TextStyle(fontSize: 16)),
          const SizedBox(height: 32),
          Center(child: FilledButton.tonal(
            onPressed: _toggleListening,
            style: FilledButton.styleFrom(shape: const CircleBorder(), padding: const EdgeInsets.all(30)),
            child: Icon(_listening ? Icons.stop : Icons.mic, size: 52),
          )),
          const SizedBox(height: 22),
          Text(_listening ? 'Listening… tap to stop' : _available ? 'Tap to speak again' : 'Tap to start', textAlign: TextAlign.center, style: const TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          Card(child: Padding(padding: const EdgeInsets.all(16), child: Text(_recognizedText))),
          const SizedBox(height: 22),
          const Text('Try saying', style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Expanded(child: ListView.builder(
            itemCount: examples.length,
            itemBuilder: (_, index) => ListTile(
              leading: const Icon(Icons.mic_none), title: Text(examples[index]),
              onTap: () => _runCommand(examples[index]),
            ),
          )),
        ]),
      ),
    );
  }
}
